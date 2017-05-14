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
import Data.Int (Int32, Int64)
import Test.Framework (defaultMain, Test)
import Lens.Family2 ((.~))
import System.IO.Temp (withSystemTempDirectory)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.ByteString.Char8 as B8

import qualified Data.Vector as V
import qualified TensorFlow.Build as TF
import qualified TensorFlow.Nodes as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF

-- | Test that one can easily determine number of elements in the tensor.
testSize :: Test
testSize = testCase "testSize" $ do
    x <- eval $ TF.size (TF.constant (TF.Shape [2, 3]) [0..5 :: Float])
    TF.Scalar (2 * 3 :: Int32) @=? x

eval :: TF.Fetchable t a => t -> IO a
eval = TF.runSession . TF.run

-- | Confirms that the original example from Python code works.
testReducedShape :: Test
testReducedShape = testCase "testReducedShape" $ do
    x <- eval $ TF.reducedShape (TF.vector [2, 3, 5, 7 :: Int64])
                                (TF.vector [1, 2 :: Int32])
    V.fromList [2, 1, 1, 7 :: Int32] @=? x

testSaveRestore :: Test
testSaveRestore = testCase "testSaveRestore" $
    withSystemTempDirectory "" $ \dirPath -> do
        let path = B8.pack $ dirPath ++ "/checkpoint"
            var :: TF.MonadBuild m => m (TF.Tensor TF.Ref Float)
            var = TF.zeroInitializedVariable' (TF.opName .~ "a")
                                        (TF.Shape [])
        TF.runSession $ do
            v <- var
            TF.assign v 134 >>= TF.run_
            TF.save path [v] >>= TF.run_
        result <- TF.runSession $ do
            v <- var
            TF.restore path v >>= TF.run_
            TF.run v
        liftIO $ TF.Scalar 134 @=? result

-- | Test that 'placeholder' is not CSE'd.
testPlaceholderCse :: Test
testPlaceholderCse = testCase "testPlaceholderCse" $ TF.runSession $ do
    p1 <- TF.placeholder []
    p2 <- TF.placeholder []
    let enc :: Float -> TF.TensorData Float
        enc n = TF.encodeTensorData [] (V.fromList [n])
    result <- TF.runWithFeeds [TF.feed p1 (enc 2), TF.feed p2 (enc 3)]
                $ p1 `TF.add` p2
    liftIO $ result @=? TF.Scalar 5

-- | Test that regular tensors can also be used for feeds, as long as they each
-- have a different name.
testScalarFeedCse :: Test
testScalarFeedCse = testCase "testScalarFeedCse" $ TF.runSession $ do
    p1 <- TF.render $ TF.scalar' (TF.opName .~ "A") 0
    -- The second op is identical to the first other than its name; make sure
    -- we don't aggressively CSE them together and prevent feeding them
    -- separately.
    p2 <- TF.render $ TF.scalar' (TF.opName .~ "B") 0
    let enc :: Float -> TF.TensorData Float
        enc n = TF.encodeTensorData [] (V.fromList [n])
    result <- TF.runWithFeeds [TF.feed p1 (enc 2), TF.feed p2 (enc 3)]
                $ p1 `TF.add` p2
    liftIO $ result @=? TF.Scalar 5

-- | See https://github.com/tensorflow/haskell/issues/92.
-- Even though we're not explicitly evaluating `f0` until the end,
-- it should hold the earlier value of the variable.
testRereadRef :: Test
testRereadRef = testCase "testReRunAssign" $ TF.runSession $ do
    w <- TF.initializedVariable 0
    f0 <- TF.run w
    TF.run_ =<< TF.assign w (TF.scalar (0.1 :: Float))
    f1 <- TF.run w
    liftIO $ (0.0, 0.1) @=? (TF.unScalar f0, TF.unScalar f1)

main :: IO ()
main = defaultMain
            [ testSaveRestore
            , testSize
            , testReducedShape
            , testPlaceholderCse
            , testScalarFeedCse
            , testRereadRef
            ]
