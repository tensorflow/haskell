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
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE OverloadedStrings #-}
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
    , render
    , renderNodeName
    , renderedNodeDefs
    , BuildT
    , Build
    , addInitializer
    , hoistBuildT
    , evalBuildT
    , runBuildT
    , asGraphDef
    , addGraphDef
    , flushInitializers
    , flushNodeBuffer
    -- * Creating and looking up Ops
    , getOrAddOp
    , addNewOp
    , renderOutput
    -- * Modifying all nodes in a Build action
    , colocateWith
    , withStateLens
    , withDevice
    , withNameScope
    , withNodeDependencies
    -- * Internal Summary related bits.
    , addSummary
    , SummaryTensor
    , collectAllSummaries
    ) where

import Control.Monad.IO.Class (MonadIO(..))
import Control.Monad.Trans.Class (MonadTrans(..))
import Control.Monad.Trans.State.Strict(StateT(..), mapStateT, evalStateT)
import Data.ByteString (ByteString)
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

import TensorFlow.Orphans ()
import TensorFlow.Output
import TensorFlow.Tensor

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

-- | Synonym for the tensors that return serialized Summary proto.
type SummaryTensor = Tensor Value ByteString

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
    , _summaries :: [SummaryTensor]
      -- ^ The tensors for summary
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

summaries :: Lens' GraphState [SummaryTensor]
summaries = lens _summaries (\g x -> g { _summaries = x })

-- | An action for building nodes in a TensorFlow graph.
-- Used to manage build state internally as part of the @Session@ monad.
newtype BuildT m a = BuildT (StateT GraphState m a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadTrans,
              MonadState GraphState)

-- | An action for building nodes in a TensorFlow graph.
type Build = BuildT Identity

-- | This is Control.Monad.Morph.hoist sans the dependency.
hoistBuildT :: (forall a . m a -> n a) -> BuildT m b -> BuildT n b
hoistBuildT f (BuildT m) = BuildT $ mapStateT f m

runBuildT :: BuildT m a -> m (a, GraphState)
runBuildT (BuildT f) = runStateT f initGraphState

evalBuildT :: Monad m => BuildT m a -> m a
evalBuildT (BuildT f) = evalStateT f initGraphState

-- | Get all the NodeDefs that have accumulated so far, and clear that buffer.
flushNodeBuffer :: Monad m => BuildT m [NodeDef]
flushNodeBuffer = do
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
addInitializer :: ControlNode -> Build ()
addInitializer (ControlNode o) = do
    i <- getOrAddOp o
    initializationNodes %= (i:)

-- | Produce a GraphDef proto representation of the nodes that are rendered in
-- the given 'Build' action.
asGraphDef :: Build a -> GraphDef
asGraphDef b = def & node .~ gs ^. nodeBuffer
  where
    gs = snd $ runIdentity $ runBuildT b

-- TODO: check against existing nodes for conflicts?
addGraphDef :: GraphDef -> Build ()
addGraphDef g = nodeBuffer <>= g ^. node

-- | Render the given op if it hasn't been rendered already, and return its
-- name.
getOrAddOp :: Op -> Build NodeName
getOrAddOp o = NodeName . (^. name) <$> resolveOp o

resolveOp :: Op -> Build NodeDef
resolveOp (Rendered n) = return n
resolveOp (Unrendered o) = do
    pending <- getPendingNode o
    uses renderedNodes (Map.lookup pending) >>= \case
        Just n -> return n
        Nothing -> addNewOpFromPending pending

-- | Add a new node for a given 'OpDef'.  This is used for making "stateful" ops
-- which are not safe to dedup (e.g, "variable" and "assign").
addNewOp :: OpDef -> Build NodeDef
addNewOp o = getPendingNode o >>= addNewOpFromPending

addNewOpFromPending :: PendingNode -> Build NodeDef
addNewOpFromPending pending = do
    nodeName <- renderPendingNode pending
    let nodeDef = pendingNodeDef pending & name .~ unNodeName nodeName
    nodeBuffer %= (nodeDef :)
    renderedNodes %= Map.insert pending nodeDef
    renderedNodeDefs %= Map.insert nodeName nodeDef
    return nodeDef

-- | Get the pending node corresponding to an OpDef, which may or may not have
-- been rendered before.  Implicitly renders all of this node's inputs.
getPendingNode :: OpDef -> Build PendingNode
getPendingNode o = do
    -- An empty string in the proto field means that no specific
    -- device is specified.
    dev <- maybe "" deviceName <$> use defaultDevice
    inputs <- mapM getInput (o ^. opInputs)
    scope <- use currentScope
    controls <- use defaultControlInputs
    let controlInputs
            = map getDep (o ^. opControlInputs ++ Set.toList controls)
    return $ PendingNode scope (o ^. opName)
            $ def & op .~ (unOpType (o ^. opType) :: Text)
                  & attr .~ _opAttrs o
                  & input .~ (inputs ++ controlInputs)
                  & device .~ dev
  where
    getInput (Output (OutputIx k) subOp)
        = (<> ":" <> Text.pack (show k)) . unNodeName <$> getOrAddOp subOp
    getDep = ("^" <>) . unNodeName

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


-- | Render an 'Output' and return a string representation for the TensorFlow
-- foreign APIs.
renderOutput :: Output -> Build Text
renderOutput (Output (OutputIx i) o) = do
    n <- getOrAddOp o
    return $ unNodeName n <> Text.pack (":" ++ show i)

-- | Modify some part of the state, run an action, and restore the state
-- after that action is done.
withStateLens :: MonadState s m => Lens' s a -> (a -> a) -> m b -> m b
withStateLens accessor f act = do
    old <- use accessor
    accessor %= f
    result <- act
    accessor .= old
    return result

-- | Set a device for all nodes rendered in the given 'Build' action
-- (unless further overridden by another use of withDevice).
withDevice :: Maybe Device -> Build a -> Build a
withDevice d = withStateLens defaultDevice (const d)

-- | Places all nodes rendered in the given 'Build' action on the same
-- device as the given Tensor (see also 'withDevice'). Make sure that
-- the action has side effects of rendering the desired tensors. A pure
-- return would not have the desired effect.
colocateWith :: forall a v b . Tensor v b -> Build a -> Build a
colocateWith t x = do
    d <- Device . (^. device) <$> resolveOp (t ^. tensorOutput . outputOp)
    withDevice (Just d) x

-- | Prepend a scope to all nodes rendered in the given 'Build' action.
withNameScope :: Text -> Build a -> Build a
withNameScope s = withStateLens currentScope (Scope s :)

-- | Add control inputs to all nodes rendered in the given 'Build' action.
withNodeDependencies :: Set NodeName -> Build a -> Build a
withNodeDependencies nodes = withStateLens defaultControlInputs (<> nodes)

-- | Render a 'Tensor', fixing its name, scope, device and control inputs from
-- the 'Build' context.  Also renders any dependencies of the 'Tensor' that
-- weren't already rendered.
--
-- This operation is idempotent; @render >=> render === render@.  However,
-- rendering a (previously un-rendered) 'Tensor' in two different contexts
-- may result in two different 'Tensor's.
render :: Tensor v a -> Build (Tensor v a)
render = tensorOutput $ outputOp $ fmap Rendered . resolveOp

-- | Render a 'Tensor' and get its node's name.
renderNodeName :: Tensor v a -> Build NodeName
renderNodeName t = getOrAddOp (t ^. tensorOutput . outputOp)

-- | Records the given summary action in Build for retrieval with
-- 'collectAllSummaries'. The summary op is required to produce a
-- Summary protocol buffer in string form. For safety, use the
-- pre-composed functions: Logging.scalarSummary and
-- Logging.histogramSummary.
addSummary :: SummaryTensor -> Build ()
addSummary t = summaries %= (t :)

-- | Retrieves the summary ops collected thus far. Typically this only
-- happens once, but if 'TensorFlow.Session.buildWithSummary' is used
-- repeatedly, the values accumulate.
collectAllSummaries :: Monad m => BuildT m [SummaryTensor]
collectAllSummaries = use summaries
