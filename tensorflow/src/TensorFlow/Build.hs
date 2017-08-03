-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE TypeFamilies #-}
module TensorFlow.Build
    ( -- * Graph node types
      ControlNode(..)
    , Unique
    -- * Ops
    , explicitName
    , implicitName
    , opDef
    , opDefWithName
    , opName
    , opType
    , opAttr
    , opInputs
    , opControlInputs
    -- * The Build monad
    , GraphState
    , renderedNodeDefs
    , BuildT
    , Build
    , MonadBuild(..)
    , addInitializer
    , hoistBuildT
    , evalBuildT
    , runBuildT
    , asGraphDef
    , addGraphDef
    , flushInitializers
    , flushNodeBuffer
    , summaries
    -- * Creating and looking up Ops
    , getOrAddOp
    , addNewOp
    , encodeOutput
    , lookupNode
    -- * Modifying all nodes in a Build action
    , withStateLens
    , withDevice
    , withNameScope
    , withNodeDependencies
    ) where

import Control.Monad.Catch (MonadThrow, MonadCatch, MonadMask)
import Control.Monad.Fix (MonadFix(..))
import Control.Monad.IO.Class (MonadIO(..))
import Control.Monad.Trans.Class (MonadTrans(..))
import Control.Monad.Trans.State.Strict(StateT(..), mapStateT, evalStateT)
import Data.Default (def)
import Data.Functor.Identity (Identity(..))
import qualified Data.Map.Strict as Map
import Data.Monoid ((<>))
import qualified Data.Set as Set
import Data.Set (Set)
import Data.String (IsString(..))
import Data.Text (Text)
import qualified Data.Text as Text
import Lens.Family2 (Lens', (.~), (^.), (&))
import Lens.Family2.State.Strict (MonadState, use, uses, (.=), (<>=), (%=))
import Lens.Family2.Unchecked (lens)
import Proto.Tensorflow.Core.Framework.Graph
    ( GraphDef
    , node
    )
import Proto.Tensorflow.Core.Framework.NodeDef
    ( NodeDef
    , attr
    , input
    , device
    , name
    , op
    )

import TensorFlow.Output

newtype Unique = Unique Int
    deriving (Eq, Ord, Enum)

--------------

implicitName :: PendingNodeName
implicitName = ImplicitName

explicitName :: Text -> PendingNodeName
explicitName = ExplicitName

newtype Scope = Scope {unScope :: Text}
    deriving (Eq, Ord, IsString)

instance Show Scope where
    show = show . unScope

opDef :: OpType -> OpDef
opDef = opDefWithName ImplicitName

opDefWithName :: PendingNodeName -> OpType -> OpDef
opDefWithName n t = OpDef
    { _opName = n
    , _opType = t
    , _opAttrs = Map.empty
    , _opInputs = []
    , _opControlInputs = []
    }

data GraphState = GraphState
    { _renderedNodes :: !(Map.Map PendingNode NodeDef)
        -- ^ Nodes which have been rendered.  Keeps track of the unique ID we
        -- assign each implicitly-named node.  Also prevents us from adding the
        -- same node (implicit or explicit) more than once to the nodeBuffer.
    , _renderedNodeDefs :: !(Map.Map NodeName NodeDef)
        -- ^ The NodeDefs of nodes which have been rendered. Used by the
        -- Gradient module to inspect the node graph.
    , _nodeBuffer :: [NodeDef]
        -- ^ A list of nodes that should be passed to TensorFlow during
        -- the next call to Session.extend (TF_ExtendGraph).
    , _nextUnique :: !Unique
        -- ^ Unique ID for the next node
    -- TODO(judahjacobson): watch for clashes between auto and user names.
    , _defaultDevice :: !(Maybe Device)
    , _currentScope :: [Scope]
    , _defaultControlInputs :: !(Set NodeName)
    , _initializationNodes  :: [NodeName]
      -- ^ The nodes to run next time a TF.run is issued, typically
      -- variable initializers.
    , _summaries :: [Output]
      -- ^ The tensors for summary (ByteString type)
    }

-- | A node definition without its final name.  Used as a key in the
-- "renderedNodes" map.
-- The NodeDef contained inside has an empty "name" field.
data PendingNode = PendingNode [Scope] !PendingNodeName !NodeDef
    deriving (Eq, Ord)

-- Returns an _incomplete_ NodeDef. The name is fixed by addNewOpFromPending.
pendingNodeDef :: PendingNode -> NodeDef
pendingNodeDef (PendingNode _ _ n) = n

initGraphState :: GraphState
initGraphState =
    GraphState Map.empty Map.empty [] (Unique 0) Nothing [] Set.empty [] []

renderedNodes :: Lens' GraphState (Map.Map PendingNode NodeDef)
renderedNodes = lens _renderedNodes (\g x -> g { _renderedNodes = x })

renderedNodeDefs :: Lens' GraphState (Map.Map NodeName NodeDef)
renderedNodeDefs = lens _renderedNodeDefs (\g x -> g { _renderedNodeDefs = x })

nodeBuffer :: Lens' GraphState [NodeDef]
nodeBuffer = lens _nodeBuffer (\g x -> g { _nodeBuffer = x })

nextUnique :: Lens' GraphState Unique
nextUnique = lens _nextUnique (\g x -> g { _nextUnique = x })

defaultDevice :: Lens' GraphState (Maybe Device)
defaultDevice = lens _defaultDevice (\g x -> g { _defaultDevice = x })

currentScope :: Lens' GraphState [Scope]
currentScope = lens _currentScope (\g x -> g { _currentScope = x })

defaultControlInputs :: Lens' GraphState (Set NodeName)
defaultControlInputs = lens _defaultControlInputs
                          (\g x -> g { _defaultControlInputs = x })

initializationNodes :: Lens' GraphState [NodeName]
initializationNodes = lens _initializationNodes (\g x -> g { _initializationNodes = x })

summaries :: Lens' GraphState [Output]
summaries = lens _summaries (\g x -> g { _summaries = x })

-- | An action for building nodes in a TensorFlow graph.
-- Used to manage build state internally as part of the @Session@ monad.
newtype BuildT m a = BuildT (StateT GraphState m a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadTrans,
              MonadState GraphState, MonadThrow, MonadCatch, MonadMask,
              MonadFix)

-- | An action for building nodes in a TensorFlow graph.
type Build = BuildT Identity

-- | This is Control.Monad.Morph.hoist sans the dependency.
hoistBuildT :: (forall a . m a -> n a) -> BuildT m b -> BuildT n b
hoistBuildT f (BuildT m) = BuildT $ mapStateT f m

runBuildT :: BuildT m a -> m (a, GraphState)
runBuildT (BuildT f) = runStateT f initGraphState

evalBuildT :: Monad m => BuildT m a -> m a
evalBuildT (BuildT f) = evalStateT f initGraphState

-- | Lift a 'Build' action into a monad, including any explicit op renderings.
class Monad m => MonadBuild m where
    build :: Build a -> m a

instance Monad m => MonadBuild (BuildT m) where
    build = hoistBuildT $ return . runIdentity

-- | Get all the NodeDefs that have accumulated so far, and clear that buffer.
flushNodeBuffer :: MonadBuild m => m [NodeDef]
flushNodeBuffer = build $ do
    ns <- use nodeBuffer
    nodeBuffer .= []
    return ns

-- | Get all the initializers that have accumulated so far, and clear
-- that buffer.
flushInitializers :: Monad m => BuildT m [NodeName]
flushInitializers = do
    ns <- use initializationNodes
    initializationNodes .= []
    return ns

-- | Registers the given node to be executed before the next
-- 'TensorFlow.Session.run'.
addInitializer :: MonadBuild m => ControlNode -> m ()
addInitializer (ControlNode i) = build $ initializationNodes %= (i:)

-- | Produce a GraphDef proto representation of the nodes that are rendered in
-- the given 'Build' action.
asGraphDef :: Build a -> GraphDef
asGraphDef b = def & node .~ gs ^. nodeBuffer
  where
    gs = snd $ runIdentity $ runBuildT b

-- TODO: check against existing nodes for conflicts?
addGraphDef :: MonadBuild m => GraphDef -> m ()
addGraphDef g = build $ nodeBuffer <>= g ^. node

-- | Render the given op if it hasn't been rendered already, and return its
-- name.
getOrAddOp :: OpDef -> Build NodeName
getOrAddOp o = do
    pending <- getPendingNode o
    uses renderedNodes (Map.lookup pending) >>= \case
        Just n -> return $ NodeName $ n ^. name
        Nothing -> addNewOpFromPending pending

lookupNode :: NodeName -> Build NodeDef
lookupNode n = uses renderedNodeDefs (Map.lookup n) >>= \case
    Just n' -> return n'
    Nothing -> error $ "lookupNode: unknown node name " ++ show n

-- | Add a new node for a given 'OpDef'.  This is used for making "stateful" ops
-- which are not safe to dedup (e.g, "variable" and "assign").
addNewOp :: OpDef -> Build NodeName
addNewOp o = getPendingNode o >>= addNewOpFromPending

addNewOpFromPending :: PendingNode -> Build NodeName
addNewOpFromPending pending = do
    nodeName <- renderPendingNode pending
    let nodeDef = pendingNodeDef pending & name .~ unNodeName nodeName
    nodeBuffer %= (nodeDef :)
    renderedNodes %= Map.insert pending nodeDef
    renderedNodeDefs %= Map.insert nodeName nodeDef
    return nodeName

-- | Get the pending node corresponding to an OpDef, which may or may not have
-- been rendered before.  Implicitly renders all of this node's inputs.
getPendingNode :: OpDef -> Build PendingNode
getPendingNode o = do
    -- An empty string in the proto field means that no specific
    -- device is specified.
    dev <- maybe "" deviceName <$> use defaultDevice
    scope <- use currentScope
    controls <- use defaultControlInputs
    let inputs = map encodeOutput (o ^. opInputs)
    let controlInputs
            = map makeDep (o ^. opControlInputs ++ Set.toList controls)
    return $ PendingNode scope (o ^. opName)
            $ def & op .~ (unOpType (o ^. opType) :: Text)
                  & attr .~ _opAttrs o
                  & input .~ (inputs ++ controlInputs)
                  & device .~ dev
  where
    makeDep = ("^" <>) . unNodeName

-- | Pick a name for a pending node.  If it has an explicit name, just use that;
-- if the name is implicit, assign a new unique name based on the op type.
renderPendingNode :: PendingNode -> Build NodeName
renderPendingNode (PendingNode scope pendingName nodeDef)
    = NodeName . (scopePrefix <>) <$> getName
  where
    scopePrefix = Text.concat $ fmap ((<> "/") . unScope) scope
    getName = case pendingName of
        ExplicitName n -> return n
        ImplicitName -> do
            u@(Unique k) <- use nextUnique
            nextUnique .= succ u
            return $ nodeDef ^. op <> "_" <> Text.pack (show k)

-- | Turn an 'Output' into a string representation for the TensorFlow
-- foreign APIs.
encodeOutput :: Output -> Text
encodeOutput (Output (OutputIx 0) n) = unNodeName n
encodeOutput (Output (OutputIx i) n) = unNodeName n <> Text.pack (':' : show i)

-- | Modify some part of the state, run an action, and restore the state
-- after that action is done.
withStateLens :: MonadBuild m => Lens' GraphState a -> (a -> a) -> m b -> m b
withStateLens accessor f act = do
    old <- build $ use accessor
    build $ accessor %= f
    result <- act
    build $ accessor .= old
    return result

-- | Set a device for all nodes rendered in the given 'Build' action
-- (unless further overridden by another use of withDevice).
withDevice :: MonadBuild m => Maybe Device -> m a -> m a
withDevice d = withStateLens defaultDevice (const d)

-- | Prepend a scope to all nodes rendered in the given 'Build' action.
withNameScope :: MonadBuild m => Text -> m a -> m a
withNameScope s = withStateLens currentScope (Scope s :)

-- | Add control inputs to all nodes rendered in the given 'Build' action.
withNodeDependencies :: MonadBuild m => Set NodeName -> m a -> m a
withNodeDependencies nodes = withStateLens defaultControlInputs (<> nodes)
