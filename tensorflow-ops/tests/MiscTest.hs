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
{-# LANGUAGE RankNTypes #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.Vector as V
import qualified TensorFlow.GenOps.Core as CoreOps

import TensorFlow.Ops
import TensorFlow.Session

-- | Test fetching multiple outputs from an op.
testMultipleOutputs :: Test
testMultipleOutputs = testCase "testMultipleOutputs" $
    runSession $ do
        (values, indices) <-
            run $ CoreOps.topKV2 (constant [1, 4] [10, 40, 20, 30]) 2
        liftIO $ [40, 30] @=? V.toList (values :: V.Vector Float)
        liftIO $ [1, 3] @=? V.toList (indices :: V.Vector Int32)

-- | Test op with variable number of inputs.
testVarargs :: Test
testVarargs = testCase "testVarargs" $
    runSession $ do
        xs <- run $ pack $ map scalar [1..8]
        liftIO $ [1..8] @=? V.toList (xs :: V.Vector Float)

main :: IO ()
main = defaultMain [ testMultipleOutputs
                   , testVarargs
                   ]
