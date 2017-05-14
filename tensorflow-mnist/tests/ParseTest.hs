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

{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64)
import Data.Text (Text)
import qualified Data.Text.IO as Text
import Lens.Family2 ((&), (.~), (^.))
import Prelude hiding (abs)
import Proto.Tensorflow.Core.Framework.Graph
    ( GraphDef(..)
    , version
    , node )
import Proto.Tensorflow.Core.Framework.NodeDef
    ( NodeDef(..)
    , op )
import System.IO as IO
import TensorFlow.Examples.MNIST.InputData
import TensorFlow.Examples.MNIST.Parse
import TensorFlow.Examples.MNIST.TrainedGraph
import TensorFlow.Build
    ( asGraphDef
    , addGraphDef
    , Build
    )
import TensorFlow.Tensor
    ( Tensor(..)
    , Ref
    , feed
    , render
    , tensorFromName
    , tensorValueFromName
    )
import TensorFlow.Ops
import TensorFlow.Session
    (runSession, run, run_, runWithFeeds, build)
import TensorFlow.Types (TensorDataType(..), Shape(..), unScalar)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?), Assertion)
import qualified Data.Vector as V

-- | Test that a file can be read and the GraphDef proto correctly parsed.
testReadMessageFromFileOrDie :: Test
testReadMessageFromFileOrDie = testCase "testReadMessageFromFileOrDie" $ do
    -- Check the function on a known well-formatted file.
    mnist <- readMessageFromFileOrDie =<< mnistPb :: IO GraphDef
    -- Simple field read.
    1 @=? mnist^.version
    -- Count the number of nodes.
    let nodes :: [NodeDef]
        nodes = mnist^.node
    100 @=? length nodes
    -- Check that the expected op is found at an arbitrary index.
    "Variable" @=? nodes!!6^.op

-- | Parse the test set for label and image data. Will only fail if the file is
--   missing or incredibly corrupt.
testReadMNIST :: Test
testReadMNIST = testCase "testReadMNIST" $ do
    imageData <- readMNISTSamples =<< testImageData
    10000 @=? length imageData
    labelData <- readMNISTLabels =<< testLabelData
    10000 @=? length labelData

testNodeName :: Text -> Tensor Build a -> Assertion
testNodeName n g = n @=? opName
  where
    opName = head (gDef^.node)^.op
    gDef = asGraphDef $ render g

testGraphDefGen :: Test
testGraphDefGen = testCase "testGraphDefGen" $ do
    -- Test the inferred operation type.
    let f0 :: Tensor Build Float
        f0 = 0
    testNodeName "Const" f0
    testNodeName "Add"  $ 1 + f0
    testNodeName "Mul"  $ 1 * f0
    testNodeName "Sub"  $ 1 - f0
    testNodeName "Abs"  $ abs f0
    testNodeName "Sign" $ signum f0
    testNodeName "Neg"  $ -f0
    -- Test the grouping.
    testNodeName "Add"  $ 1 + f0 * 2
    testNodeName "Add"  $ 1 + (f0 * 2)
    testNodeName "Mul"  $ (1 + f0) * 2

-- | Convert a simple graph to GraphDef, load it, run it, and check the output.
testGraphDefExec :: Test
testGraphDefExec = testCase "testGraphDefExec" $ do
    let graphDef = asGraphDef $ render $ scalar (5 :: Float) * 10
    runSession $ do
        addGraphDef graphDef
        x <- run $ tensorValueFromName "Mul_2"
        liftIO $ (50 :: Float) @=? unScalar x

-- | Load MNIST from a GraphDef and the weights from a checkpoint and run on
--   sample data.
testMNISTExec :: Test
testMNISTExec = testCase "testMNISTExec" $ do
    -- Switch to unicode to enable pretty printing of MNIST digits.
    IO.hSetEncoding IO.stdout IO.utf8

    -- Parse the Graph definition, samples, & labels from files.
    mnist <- readMessageFromFileOrDie =<< mnistPb :: IO GraphDef
    mnistSamples <- readMNISTSamples =<< testImageData
    mnistLabels <- readMNISTLabels =<< testLabelData

    -- Select a sample to run on and convert it into a TensorData of Floats.
    let idx = 12
        sample :: MNIST
        sample = mnistSamples !! idx
        label = mnistLabels !! idx
        tensorSample = encodeTensorData (Shape [1,784]) floatSample
          where
            floatSample :: V.Vector Float
            floatSample = V.map fromIntegral sample
    Text.putStrLn $ drawMNIST sample

    -- Execute the graph on the sample data.
    runSession $ do
        -- The version of this session is 0, but the version of the graph is 1.
        -- Change the graph version to 0 so they're compatible.
        build $ addGraphDef $ mnist & version .~ 0
        -- Define nodes that restore saved weights and biases.
        let bias, wts :: Tensor Ref Float
            bias = tensorFromName "Variable"
            wts = tensorFromName "weights"
        wtsCkptPath <- liftIO wtsCkpt
        biasCkptPath <- liftIO biasCkpt
        -- Run those restoring nodes on the graph in the current session.
        run_ =<< (sequence :: Monad m => [m a] -> m [a])
                        [ restore wtsCkptPath wts
                        , restoreFromName biasCkptPath "bias" bias
                        ]
        -- Encode the expected sample data as one-hot data.
        let ty = encodeTensorData [10] oneHotLabels
              where oneHotLabels = V.replicate 10 (0 :: Float) V.// updates
                    updates = [(fromIntegral label, 1)]
        let feeds = [ feed (tensorValueFromName "x-input") tensorSample
                    , feed (tensorValueFromName "y-input") ty
                    ]
        -- Run the graph with the input feeds and read the ArgMax'd result from
        -- the test (not training) side of the evaluation.
        x <- runWithFeeds feeds $ tensorValueFromName "test/ArgMax"
        -- Print the trained model's predicted outcome.
        liftIO $ putStrLn $ "Expectation: " ++ show label ++ "\n"
                         ++ "Prediction:  " ++ show (unScalar x :: Int64)
        -- Check whether the prediction matches the expectation.
        liftIO $ (fromInteger . toInteger $ label :: Int64) @=? unScalar x

main :: IO ()
main = defaultMain
            [ testReadMessageFromFileOrDie
            , testReadMNIST
            , testGraphDefGen
            , testGraphDefExec
            , testMNISTExec ]
