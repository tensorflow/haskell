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

-- | TensorBoard Summary generation. Provides type safe wrappers around raw
-- string emitting CoreOps.
--
-- Example use:
--
-- > -- Call summary functions while constructing the graph.
-- > createModel = do
-- >   loss <- -- ...
-- >   TF.scalarSummary loss
-- >
-- > -- Write summaries to an EventWriter.
-- > train = TF.withEventWriter "/path/to/logs" $ \eventWriter -> do
-- >     summaryTensor <- TF.build TF.allSummaries
-- >     forM_ [1..] $ \step -> do
-- >         if (step % 100 == 0)
-- >             then do
-- >                 ((), summaryBytes) <- TF.run (trainStep, summaryTensor)
-- >                 let summary = decodeMessageOrDie (TF.unScalar summaryBytes)
-- >                 TF.logSummary eventWriter step summary
-- >             else TF.run_ trainStep

{-# LANGUAGE TypeOperators #-}

module TensorFlow.Logging
    ( EventWriter
    , withEventWriter
    , logEvent
    , logGraph
    , logSummary
    , SummaryTensor
    , histogramSummary
    , scalarSummary
    , mergeAllSummaries
    ) where

import Control.Concurrent (forkFinally)
import Control.Concurrent.MVar (MVar, newEmptyMVar, readMVar, putMVar)
import Control.Concurrent.STM (atomically)
import Control.Concurrent.STM.TBMQueue (TBMQueue, newTBMQueueIO, closeTBMQueue, writeTBMQueue)
import Control.Monad.Catch (MonadMask, bracket)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Resource (runResourceT)
import Data.ByteString (ByteString)
import Data.Conduit ((=$=))
import Data.Conduit.TQueue (sourceTBMQueue)
import Data.Default (def)
import Data.Int (Int64)
import Data.ProtoLens (encodeMessage)
import Data.Time.Clock (getCurrentTime)
import Data.Time.Clock.POSIX (utcTimeToPOSIXSeconds)
import Lens.Family2 ((.~), (&))
import Network.HostName (getHostName)
import Proto.Tensorflow.Core.Framework.Summary (Summary)
import Proto.Tensorflow.Core.Util.Event (Event, fileVersion, graphDef, step, summary, wallTime)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import TensorFlow.Build (MonadBuild, Build, asGraphDef)
import TensorFlow.Ops (scalar)
import TensorFlow.Records.Conduit (sinkTFRecords)
import TensorFlow.Tensor (Tensor, render, SummaryTensor, addSummary, collectAllSummaries)
import TensorFlow.Types (TensorType, type(/=))
import Text.Printf (printf)
import qualified Data.ByteString.Lazy as L
import qualified Data.Conduit as Conduit
import qualified Data.Conduit.List as Conduit
import qualified Data.Text as T
import qualified TensorFlow.GenOps.Core as CoreOps

-- | Handle for logging TensorBoard events safely from multiple threads.
data EventWriter = EventWriter (TBMQueue Event) (MVar ())

-- | Writes Event protocol buffers to event files.
withEventWriter ::
    (MonadIO m, MonadMask m)
    => FilePath
    -- ^ logdir. Local filesystem directory where event file will be written.
    -> (EventWriter -> m a)
    -> m a
withEventWriter logdir =
    bracket (liftIO (newEventWriter logdir)) (liftIO . closeEventWriter)

newEventWriter :: FilePath -> IO EventWriter
newEventWriter logdir = do
    createDirectoryIfMissing True logdir
    t <- doubleWallTime
    hostname <- getHostName
    let filename = printf (logdir </> "events.out.tfevents.%010d.%s")
                          (truncate t :: Integer) hostname
    -- Asynchronously consume events from a queue.
    -- We use a bounded queue to ensure the producer doesn't get too far ahead
    -- of the consumer. The buffer size was picked arbitrarily.
    q <- newTBMQueueIO 1024
    -- Use an MVar to signal that the worker thread has completed.
    done <- newEmptyMVar
    let writer = EventWriter q done
        consumeQueue = runResourceT $ Conduit.runConduit $
            sourceTBMQueue q
            =$= Conduit.map (L.fromStrict . encodeMessage)
            =$= sinkTFRecords filename
    _ <- forkFinally consumeQueue (\_ -> putMVar done ())
    logEvent writer $ def & wallTime .~ t
                          & fileVersion .~ T.pack "brain.Event:2"
    return writer

closeEventWriter :: EventWriter -> IO ()
closeEventWriter (EventWriter q done) =
    atomically (closeTBMQueue q) >> readMVar done

-- | Logs the given Event protocol buffer.
logEvent :: MonadIO m => EventWriter -> Event -> m ()
logEvent (EventWriter q _) pb = liftIO (atomically (writeTBMQueue q pb))

-- | Logs the graph for the given 'Build' action.
logGraph :: MonadIO m => EventWriter -> Build a -> m ()
logGraph writer build = do
  let graph = asGraphDef build
      graphBytes = encodeMessage graph
      graphEvent = (def :: Event) & graphDef .~ graphBytes
  logEvent writer graphEvent

-- | Logs the given Summary event with an optional global step (use 0 if not
-- applicable).
logSummary :: MonadIO m => EventWriter -> Int64 -> Summary -> m ()
logSummary writer step' summaryProto = do
    t <- liftIO doubleWallTime
    logEvent writer (def & wallTime .~ t
                         & step .~ step'
                         & summary .~ summaryProto
                    )


-- Number of seconds since epoch.
doubleWallTime :: IO Double
doubleWallTime = asDouble <$> getCurrentTime
    where asDouble t = fromRational (toRational (utcTimeToPOSIXSeconds t))

-- | Adds a 'CoreOps.histogramSummary' node. The tag argument is intentionally
-- limited to a single value for simplicity.
histogramSummary ::
    (MonadBuild m, TensorType t, t /= ByteString, t /= Bool)
     -- OneOf '[Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t)
    => ByteString -> Tensor v t -> m ()
histogramSummary tag = addSummary . CoreOps.histogramSummary (scalar tag)

-- | Adds a 'CoreOps.scalarSummary' node.
scalarSummary ::
    (TensorType t, t /= ByteString, t /= Bool, MonadBuild m)
    -- (TensorType t,
    --  OneOf '[Int16, Int32, Int64, Int8, Word16, Word8, Double, Float] t)
    => ByteString -> Tensor v t -> m ()
scalarSummary tag = addSummary . CoreOps.scalarSummary (scalar tag)

-- | Merge all summaries accumulated in the 'Build' into one summary.
mergeAllSummaries :: MonadBuild m => m SummaryTensor
mergeAllSummaries = collectAllSummaries >>= render . CoreOps.mergeSummary
