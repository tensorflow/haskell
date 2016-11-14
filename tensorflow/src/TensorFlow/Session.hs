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
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module TensorFlow.Session (
    Session,
    SessionOption,
    sessionConfig,
    sessionTarget,
    runSession,
    runSessionWithOptions,
    build,
    buildAnd,
    buildWithSummary,
    extend,
    addGraphDef,
    run,
    runWithFeeds,
    run_,
    runWithFeeds_,
    asyncProdNodes,
    ) where

import Control.Monad (forever, unless, void)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Reader (ReaderT(..), ask, asks)
import Data.ByteString (ByteString)
import Data.Functor.Identity (runIdentity)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Set (Set)
import Data.Text.Encoding (encodeUtf8)
import Data.ProtoLens (def)
import Lens.Family2 ((&), (.~))
import Proto.Tensorflow.Core.Framework.Graph (node)
import Proto.Tensorflow.Core.Protobuf.Config (ConfigProto)

import TensorFlow.Build
import qualified TensorFlow.Internal.FFI as FFI
import qualified TensorFlow.Internal.Raw as Raw
import TensorFlow.Nodes
import TensorFlow.Output (NodeName, unNodeName)
import TensorFlow.Tensor

-- Common state threaded through the session.
data SessionState
    = SessionState {
          rawSession :: FFI.Session
        , asyncCollector :: IO () -> IO ()
          -- ^ Starts the given action concurrently.
        }

newtype Session a
    = Session (ReaderT SessionState (BuildT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO)

-- | Run 'Session' actions in a new TensorFlow session.
runSession :: Session a -> IO a
runSession = runSessionWithOptions []

-- | Setting of an option for the session (see 'runSessionWithOptions').
-- Opaque value created via 'sessionConfig' and 'sessionTarget'.
newtype SessionOption =
    SessionOption { unSesssionOption :: Raw.SessionOptions -> IO () }

-- | Target can be: "local", ip:port, host:port.
-- The set of supported factories depends on the linked in libraries.
-- REQUIRES "//learning/brain/public:tensorflow_remote" dependency for the binary.
sessionTarget :: ByteString -> SessionOption
sessionTarget = SessionOption . FFI.setSessionTarget

-- | Uses the specified config for the created session.
sessionConfig :: ConfigProto -> SessionOption
sessionConfig = SessionOption . FFI.setSessionConfig

-- | Run 'Session' actions in a new TensorFlow session created with
-- the given option setter actions ('sessionTarget', 'sessionConfig').
runSessionWithOptions :: [SessionOption] -> Session a -> IO a
runSessionWithOptions options (Session m) =
    FFI.withSession applyOptions $
        \as rs -> evalBuildT (runReaderT m (SessionState rs as))
  where applyOptions opt = mapM_ (`unSesssionOption` opt) options

-- | Lift a 'Build' action into a 'Session', including any explicit op
-- renderings.
build :: Build a -> Session a
build = Session . lift . hoistBuildT (return . runIdentity)

-- | Lift a 'Build' action into a 'Session', including any explicit op
-- renderings. Returns the merged summary ops which can be used for
-- logging, see 'TensorFlow.Logging.build' for a convenient wrapper.
buildWithSummary :: forall a . Build a -> Session (a, [SummaryTensor])
buildWithSummary b = Session $ lift $ (,) <$> v <*> collectAllSummaries
  where v :: BuildT IO a
        v = hoistBuildT (return . runIdentity) b

-- | Add all pending rendered nodes to the TensorFlow graph and runs
-- any pending initializers.
--
-- Note that run, runWithFeeds, etc. will all call this function implicitly.
extend :: Session ()
extend = do
    let withSessionWhen vs action =
            unless (null vs) $ Session (asks rawSession) >>= action
    nodesToExtend <- build flushNodeBuffer
    withSessionWhen nodesToExtend $ \session ->
        liftIO $ FFI.extendGraph session
               $ def & node .~ nodesToExtend
    -- Now that all the nodes are created, run the initializers.
    initializers <- build flushInitializers
    withSessionWhen initializers $ \session ->
        void $ liftIO $ FFI.run session [] [] (toNodeNames initializers)

-- | Helper combinator for doing something with the result of a 'Build' action.
-- Example usage:
--
-- > buildAnd run :: Fetchable t a => Build t -> Session a
buildAnd :: (a -> Session b) -> Build a -> Session b
buildAnd f m = build m >>= f

-- | Run a subgraph 't', rendering any dependent nodes that aren't already
-- rendered, and fetch the corresponding values for 'a'.
run :: Fetchable t a => t -> Session a
run = runWithFeeds []

-- | Run a subgraph 't', rendering any dependent nodes that aren't already
-- rendered, feed the given input values, and fetch the corresponding result
-- values for 'a'.
runWithFeeds :: Fetchable t a => [Feed] -> t -> Session a
runWithFeeds feeds t = do
    ns <- build $ getNodes t
    -- Note that this call to "fetch" shouldn't affect the following "extend"
    -- call, since all nodes in t and its inputs/deps will be rendered by the
    -- above call to getNodes.
    fetch <- build $ getFetch t
    runFetchWithFeeds feeds ns fetch

runFetchWithFeeds :: [Feed] -> Set NodeName -> Fetch a -> Session a
runFetchWithFeeds feeds target (Fetch fetch restore) = do
    extend
    feeds' <- build $ fixFeeds feeds
    let fetchNames = encodeUtf8 <$> Set.toList fetch
        targetNames = toNodeNames $ Set.toList target
    session <- Session (asks rawSession)
    runResult <- liftIO $ FFI.run session
                                  feeds'
                                  fetchNames
                                  targetNames
    let resultTensorsMap = Map.fromList $ zip (Set.toList fetch) runResult
    return $ restore resultTensorsMap

toNodeNames :: [NodeName] -> [ByteString]
toNodeNames = map (encodeUtf8 . unNodeName)

-- | Run a subgraph 't', rendering and extending any dependent nodes that aren't
-- already rendered.  This behaves like 'run' except that it doesn't do any
-- fetches.
run_ :: Nodes t => t -> Session ()
run_ = runWithFeeds_ []

-- | Run a subgraph 't', rendering any dependent nodes that aren't already
-- rendered, feed the given input values, and fetch the corresponding result
-- values for 'a'.  This behaves like 'runWithFeeds' except that it doesn't do
-- any fetches.
runWithFeeds_ :: Nodes t => [Feed] -> t -> Session ()
runWithFeeds_ feeds t = do
    ns <- build $ getNodes t
    runFetchWithFeeds feeds ns (pure ())

fixFeeds :: [Feed] -> Build [(ByteString, FFI.TensorData)]
fixFeeds = mapM $ \(Feed o d) -> (,d) . encodeUtf8 <$> renderOutput o

-- | Starts a concurrent thread which evaluates the given Nodes
-- forever until runSession exits or an exception occurs. Graph
-- extension happens synchronously, but the resultant run proceeds as
-- a separate thread.
asyncProdNodes :: Nodes t
                  => t  -- ^ Node to evaluate concurrently.
                  -> Session ()
asyncProdNodes nodes = do
    target <- build (getNodes nodes)
    extend
    let targetNames = toNodeNames $ Set.toList target
    state <- Session ask
    let loop = forever (void (FFI.run (rawSession state) [] [] targetNames))
    liftIO (asyncCollector state loop)
