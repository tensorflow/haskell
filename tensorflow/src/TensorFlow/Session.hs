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
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module TensorFlow.Session (
    Session,
    Options,
    sessionConfig,
    sessionTarget,
    sessionTracer,
    runSession,
    runSessionWithOptions,
    MonadBuild(..),
    extend,
    addGraphDef,
    run,
    runWithFeeds,
    run_,
    runWithFeeds_,
    asyncProdNodes,
    ) where

import Control.Monad (forever, unless, void)
import Control.Monad.Catch (MonadThrow, MonadCatch, MonadMask)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Reader (ReaderT(..), ask, asks)
import Data.ByteString (ByteString)
import Data.Default (Default, def)
import Data.Monoid ((<>))
import Data.ProtoLens (showMessage)
import Data.Set (Set)
import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 (Lens', (^.), (&), (.~))
import Lens.Family2.Unchecked (lens)
import Proto.Tensorflow.Core.Framework.Graph (GraphDef, node)
import Proto.Tensorflow.Core.Protobuf.Config (ConfigProto)
import TensorFlow.Build
import TensorFlow.Nodes
import TensorFlow.Output (NodeName, unNodeName)
import TensorFlow.Tensor

import qualified Data.ByteString.Builder as Builder
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified TensorFlow.Internal.FFI as FFI

-- | An action for logging.
type Tracer = Builder.Builder -> IO ()

-- Common state threaded through the session.
data SessionState
    = SessionState {
          rawSession :: FFI.Session
        , asyncCollector :: IO () -> IO ()
          -- ^ Starts the given action concurrently.
        , tracer :: Tracer
        }

newtype Session a
    = Session (ReaderT SessionState (BuildT IO) a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadThrow, MonadCatch,
              MonadMask)

-- | Run 'Session' actions in a new TensorFlow session.
runSession :: Session a -> IO a
runSession = runSessionWithOptions def

-- | Customization for session. Use the lenses to update:
-- 'sessionTarget', 'sessionTracer', 'sessionConfig'.
data Options = Options
    { _sessionTarget :: ByteString
    , _sessionConfig :: ConfigProto
    , _sessionTracer :: Tracer
    }

instance Default Options where
    def = Options
          { _sessionTarget = ""
          , _sessionConfig = def
          , _sessionTracer = const (return ())
          }

-- | Target can be: "local", ip:port, host:port.
-- The set of supported factories depends on the linked in libraries.
sessionTarget :: Lens' Options ByteString
sessionTarget = lens _sessionTarget (\g x -> g { _sessionTarget = x })

-- | Uses the specified config for the created session.
sessionConfig :: Lens' Options ConfigProto
sessionConfig = lens _sessionConfig (\g x -> g { _sessionConfig = x })

-- | Uses the given logger to monitor session progress.
sessionTracer :: Lens' Options Tracer
sessionTracer = lens _sessionTracer (\g x -> g { _sessionTracer = x })

-- | Run 'Session' actions in a new TensorFlow session created with
-- the given option setter actions ('sessionTarget', 'sessionConfig').
runSessionWithOptions :: Options -> Session a -> IO a
runSessionWithOptions options (Session m) =
    FFI.withSession applyOptions $
        \as rs ->
            let initState = SessionState rs as (options ^. sessionTracer)
            in evalBuildT (runReaderT m initState)
  where applyOptions opt = do
            FFI.setSessionTarget (options ^. sessionTarget) opt
            FFI.setSessionConfig (options ^. sessionConfig) opt

instance MonadBuild Session where
    build = Session . lift . build

-- | Add all pending rendered nodes to the TensorFlow graph and runs
-- any pending initializers.
--
-- Note that run, runWithFeeds, etc. will all call this function implicitly.
extend :: Session ()
extend = do
    session <- Session (asks rawSession)
    trace <- Session (asks tracer)
    nodesToExtend <- build flushNodeBuffer
    unless (null nodesToExtend) $ liftIO $ do
        let graphDef = (def :: GraphDef) & node .~ nodesToExtend
        trace ("Session.extend " <> Builder.string8 (showMessage graphDef))
        FFI.extendGraph session graphDef
    -- Now that all the nodes are created, run the initializers.
    initializers <- build flushInitializers
    unless (null initializers) $
        void $ liftIO $ FFI.run session [] [] (toNodeNames initializers)

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
    let feeds' = fixFeeds feeds
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

fixFeeds :: [Feed] -> [(ByteString, FFI.TensorData)]
fixFeeds = map $ \(Feed o d) -> (encodeUtf8 $ encodeOutput o, d)

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
