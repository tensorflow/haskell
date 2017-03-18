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
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Data.Int (Int32)
import Data.List (sort)
import Data.ProtoLens.TextFormat (showMessage)
import Google.Test (googleTest)
import Lens.Family2 ((^..))
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.Vector as V

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (max)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF

import Proto.Tensorflow.Core.Framework.Graph (node)
import Proto.Tensorflow.Core.Framework.NodeDef (op)

testGradientSimple :: Test
testGradientSimple = testCase "testGradientSimple" $ do
    let grads = do
                x <- TF.render $ TF.scalar (3 :: Float)
                b <- TF.render $ TF.scalar (4 :: Float)
                let y = x `TF.mul` x `TF.add` b
                TF.gradients y [x, b]
    -- Assert that the gradients are right.
    [dx, db] <- TF.runSession $ grads >>= TF.run
    6 @=? TF.unScalar dx
    1 @=? TF.unScalar db
    -- Assert that the graph has the expected ops.
    let graphDef = TF.asGraphDef grads
    putStrLn $ showMessage graphDef
    let ops = graphDef ^.. node . traverse . op
        expected = [ "Const"
                   , "Mul"
                   , "Const"
                   , "Add"
                     -- Default output gradient of y.
                   , "Shape"
                   , "Const"
                   , "Fill"
                     -- Add gradient.
                   , "Shape"
                   , "Shape"
                   , "BroadcastGradientArgs"
                   , "Sum"
                   , "Sum"
                   , "Reshape"
                   , "Reshape"
                     -- Mul gradient.
                   , "Shape"
                   -- This Op gets dedup'd because the inputs are the same.
                   -- TODO(fmayle): The same would happen to the Mul and Sum ops
                   -- below if the gradient function didn't multiply one as
                   -- 'dz * y' and the other as 'x * dz'. We could change the
                   -- order, but I'm going to keep it the same as the python
                   -- version for now.
                   --
                   -- , "Shape"
                   , "BroadcastGradientArgs"
                   , "Mul"
                   , "Mul"
                   , "Sum"
                   , "Sum"
                   , "Reshape"
                   , "Reshape"
                     -- AddN to combine x's output gradients.
                   , "AddN"
                   ]
    sort expected @=? sort ops

testGradientDisconnected :: Test
testGradientDisconnected = testCase "testGradientDisconnected" $ do
    let grads = do
            x <- TF.render $ TF.scalar (3 :: Float)
            b <- TF.render $ TF.scalar (4 :: Float)
            TF.gradients x [x, b]
    -- Assert that the gradients are right.
    [dx, db] <- TF.runSession $ grads >>= TF.run
    1 @=? TF.unScalar dx
    0 @=? TF.unScalar db
    -- Assert that the graph has the expected ops.
    let graphDef = TF.asGraphDef grads
    putStrLn $ showMessage graphDef
    let ops = graphDef ^.. node . traverse . op
        expected = [ "Const"
                   , "Const"
                     -- Default output gradient of x.
                   , "Shape"
                   , "Const"
                   , "Fill"
                     -- Default output gradient of b.
                   , "ZerosLike"
                   ]
    sort expected @=? sort ops


-- Test that identical "stateful" ops work with createGraph.
testCreateGraphStateful :: Test
testCreateGraphStateful = testCase "testCreateGraphStateful" $ do
    [dx, dy] <- TF.runSession $ do
        let shape = TF.constant (TF.Shape [1]) [1]
        x :: TF.Tensor TF.Value Float <- TF.truncatedNormal shape
        y :: TF.Tensor TF.Value Float <- TF.truncatedNormal shape
        TF.gradients (TF.expr x + TF.expr y * 3) [x, y] >>= TF.run
    -- If this test fails, it will likely be caused by an exception within
    -- `TF.gradients`. These asserts are extra.
    1 @=? TF.unScalar dx
    3 @=? TF.unScalar dy


-- Test that name scopes work with createGraph.
testCreateGraphNameScopes :: Test
testCreateGraphNameScopes = testCase "testCreateGraphNameScopes" $ do
    [dx] <- TF.runSession $ do
        let shape = TF.constant (TF.Shape [1]) [1]
        x :: TF.Tensor TF.Value Float <-
            TF.withNameScope "foo" (TF.truncatedNormal shape)
        TF.gradients x [x] >>= TF.run
    -- If this test fails, it will likely be caused by an exception within
    -- `TF.gradients`. This assert is extra.
    1 @=? TF.unScalar dx


-- Test that createGraph can handle graphs with diamond shapes.
testDiamond :: Test
testDiamond = testCase "testDiamond" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1]
        let y = x `TF.mul` x
            z = y*y
        TF.gradients z [x] >>= TF.run
    (4 :: Float) @=? TF.unScalar dx


testMaxGradient :: Test
testMaxGradient = testCase "testMaxGradient" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1, 2, 3, 0, 1 :: Float]
        let y = TF.max x (0 :: TF.Tensor TF.Build Int32)
        TF.gradients y [x] >>= TF.run
    V.fromList [0, 0, 1, 0, 0 :: Float] @=? dx


main :: IO ()
main = googleTest [ testGradientSimple
                  , testGradientDisconnected
                  , testCreateGraphStateful
                  , testCreateGraphNameScopes
                  , testDiamond
                  , testMaxGradient
                  ]
