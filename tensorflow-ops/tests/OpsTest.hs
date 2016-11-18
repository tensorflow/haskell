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

{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32, Int64)
import Google.Test (googleTest)
import System.IO.Temp (withSystemTempDirectory)
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.ByteString.Char8 as B8

import qualified Data.Vector as V
import qualified TensorFlow.Build as TF
import qualified TensorFlow.ControlFlow as TF
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
eval = TF.runSession . TF.buildAnd TF.run . return

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
            var :: TF.Build (TF.Tensor TF.Ref Float)
            var = TF.render =<<
                  TF.named "a" <$> TF.zeroInitializedVariable (TF.Shape [])
        TF.runSession $ do
            v <- TF.build var
            TF.buildAnd TF.run_ $ TF.assign v 134
            TF.buildAnd TF.run_ $ TF.save path [v]
        result <- TF.runSession $ do
            v <- TF.build var
            TF.buildAnd TF.run_ $ TF.restore path v
            TF.run v
        liftIO $ TF.Scalar 134 @=? result


main :: IO ()
main = googleTest [ testSaveRestore
                  , testSize
                  , testReducedShape
                  ]
