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

-- | Tests for TensorFlow.Logging.
module Main where

import Control.Monad.Trans.Resource (runResourceT)
import Data.Conduit ((=$=))
import Data.Default (def)
import Data.List ((\\))
import Data.ProtoLens (decodeMessageOrDie)
import Lens.Family2 ((^.), (.~), (&))
import Proto.Tensorflow.Core.Util.Event (Event, fileVersion, step)
import System.Directory (getDirectoryContents)
import System.FilePath ((</>))
import System.IO.Temp (withSystemTempDirectory)
import TensorFlow.Logging (withEventWriter, logEvent)
import TensorFlow.Records.Conduit (sourceTFRecords)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit (assertBool, assertEqual)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Conduit as Conduit
import qualified Data.Conduit.List as Conduit
import qualified Data.Text as T

-- TODO: This has been added to System.Directory in newer versions.
listDirectory :: String -> IO [String]
listDirectory dir = (\\ [".", ".."]) <$> getDirectoryContents dir

testEventWriter :: Test
testEventWriter = testCase "EventWriter" $
    withSystemTempDirectory "event_writer_logs" $ \dir -> do
        assertEqual "No file before" [] =<< listDirectory dir
        let expected = [ (def :: Event) & step .~ 10
                       , def & step .~ 222
                       , def & step .~ 8
                       ]
        withEventWriter dir $ \eventWriter ->
            mapM_ (logEvent eventWriter) expected
        files <- listDirectory dir
        assertEqual "One file exists after" 1 (length files)
        records <- runResourceT $ Conduit.runConduit $
            sourceTFRecords (dir </> head files) =$= Conduit.consume
        assertBool "File is not empty" (not (null records))
        let (header:body) = decodeMessageOrDie . BL.toStrict <$> records
        assertEqual "Header has expected version"
                    (T.pack "brain.Event:2") (header ^. fileVersion)
        assertEqual "Body has expected records" expected body

main :: IO ()
main = defaultMain [ testEventWriter
                   ]
