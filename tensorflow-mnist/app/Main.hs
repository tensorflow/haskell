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

import Control.Monad (zipWithM, when, forM, forM_)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32, Int64)
import qualified Data.Text.IO as T
import qualified Data.Vector as V

import qualified TensorFlow.ControlFlow as TF
import qualified TensorFlow.Build as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.Gradient as TF

import TensorFlow.Examples.MNIST.InputData
import TensorFlow.Examples.MNIST.Parse

numPixels = 28^2 :: Int64
numLabels = 10 :: Int64

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Value Float)
randomParam width (TF.Shape shape) =
    (* stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

-- Types must match due to model structure (sparseToDense requires
-- index types to match)
type LabelType = Int32
type BatchSize = Int32

-- | Convert scalar labels to one-hot vectors.
labelClasses :: TF.Tensor TF.Value LabelType
             -> LabelType
             -> BatchSize
             -> TF.Tensor TF.Value Float
labelClasses labels numClasses batchSize =
    let indices = TF.range 0 (TF.scalar batchSize) 1
        concated = TF.concat 1 [TF.expandDims indices 1, TF.expandDims labels 1]
    in TF.sparseToDense concated
       (TF.constant [2] [batchSize, numClasses])
       1 {- ON value -}
       0 {- default (OFF) value -}

-- | Fraction of elements that differ between two vectors.
errorRate :: Eq a => V.Vector a -> V.Vector a -> Double
errorRate xs ys = fromIntegral (len - numCorrect) / fromIntegral len
  where
    numCorrect = V.length $ V.filter id $ V.zipWith (==) xs ys
    len = V.length xs

data Model = Model {
      train :: TF.TensorData Float  -- ^ images
            -> TF.TensorData LabelType
            -> TF.Session ()
    , infer :: TF.TensorData Float  -- ^ images
            -> TF.Session (V.Vector LabelType)  -- ^ predictions
    }

createModel :: Int64 -> TF.Build Model
createModel batchSize = do
    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]
    -- Hidden layer.
    let numUnits = 500
    hiddenWeights <-
        TF.initializedVariable =<< randomParam numPixels [numPixels, numUnits]
    hiddenBiases <- TF.zeroInitializedVariable [numUnits]
    let hiddenZ = (images `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
    let hidden = TF.relu hiddenZ
    -- Logits.
    logitWeights <-
        TF.initializedVariable =<< randomParam numUnits [numUnits, numLabels]
    logitBiases <- TF.zeroInitializedVariable [numLabels]
    let logits = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases
    predict <- TF.render $ TF.cast $
               TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType))

    -- Create training action.
    labels <- TF.placeholder [batchSize]
    let labelVecs = labelClasses labels 10 (fromIntegral batchSize)
        loss = fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
        params = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]
    grads <- TF.gradients loss params

    let lr = TF.scalar $ 0.001 / fromIntegral batchSize
        applyGrad param grad
            = TF.assign param $ param `TF.sub` (lr * grad)
    trainStep <- TF.group =<< zipWithM applyGrad params grads

    return Model {
        train = \imFeed lFeed -> TF.runWithFeeds_ [
              TF.feed images imFeed
            , TF.feed labels lFeed
            ] trainStep
        , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
        }

main = TF.runSession $ do
    -- Read training and test data.
    trainingImages <- liftIO (readMNISTSamples =<< trainingImageData)
    trainingLabels <- liftIO (readMNISTLabels =<< trainingLabelData)
    testImages <- liftIO (readMNISTSamples =<< testImageData)
    testLabels <- liftIO (readMNISTLabels =<< testLabelData)

    let batchSize = 100 :: Int64

    -- Create the model.
    model <- TF.build $ createModel batchSize

    -- Helpers for generate batches.
    let selectBatch i xs = take size $ drop (i * size) $ cycle xs
          where size = fromIntegral batchSize
    let getImageBatch i xs = TF.encodeTensorData
            [batchSize, numPixels]
            $ fromIntegral <$> mconcat (selectBatch i xs)
    let getExpectedLabelBatch i xs =
            fromIntegral <$> V.fromList (selectBatch i xs)

    -- Train.
    forM_ ([0..1000] :: [Int]) $ \i -> do
        let images = getImageBatch i trainingImages
            labels = getExpectedLabelBatch i trainingLabels
        train model images (TF.encodeTensorData [batchSize] labels)
        when (i `mod` 100 == 0) $ do
            preds <- infer model images
            liftIO $ putStrLn $
                "training error " ++ show (errorRate preds labels * 100)
    liftIO $ putStrLn ""

    -- Test.
    let numTestBatches = length testImages `div` fromIntegral batchSize
    testPreds <- fmap mconcat $ forM [0..numTestBatches] $ \i -> do
        infer model (getImageBatch i testImages)
    let testExpected = fromIntegral <$> V.fromList testLabels
    liftIO $ putStrLn $
        "test error " ++ show (errorRate testPreds testExpected * 100)

    -- Show some predictions.
    liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
        putStrLn ""
        T.putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "expected " ++ show (testLabels !! i)
        putStrLn $ "     got " ++ show (testPreds V.! i)
