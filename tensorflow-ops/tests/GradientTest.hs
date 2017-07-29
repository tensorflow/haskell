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
{-# LANGUAGE FlexibleContexts #-}

import Data.Int (Int32, Int64)
import Data.List (sort)
import qualified Data.List as List
import Data.ProtoLens.TextFormat (showMessage)
import Test.Framework (defaultMain, Test)
import Lens.Family2 ((^..), (.~))

import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?), assertEqual)
import qualified Data.Vector as V
import System.Random (randomIO, randomRIO)
import Control.Monad(forM_, replicateM, zipWithM)
import Control.Monad.IO.Class (liftIO)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (max, tile, maximum)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF hiding (zeroInitializedVariable)
import qualified TensorFlow.Output as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.Variable as TF

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


testAddNGradient :: Test
testAddNGradient = testCase "testAddNGradient" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1, 2, 0 :: Float]
        let y = TF.addN [x, x]
        TF.gradients y [x] >>= TF.run
    V.fromList [2, 2, 2 :: Float] @=? dx


testMaxGradient :: Test
testMaxGradient = testCase "testMaxGradient" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [1, 2, 3, 0, 1 :: Float]
        let y = TF.max x (0 :: TF.Tensor TF.Build Int32)
        TF.gradients y [x] >>= TF.run
    V.fromList [0, 0, 1, 0, 0 :: Float] @=? dx

testConcatGradient :: Test
testConcatGradient = testCase "testConcatGradient" $ do
    [dv,dv'] <- TF.runSession $ do
        v  <- TF.render $ TF.vector [1 :: Float]
        v' <- TF.render $ TF.vector [2 :: Float]
        let y = TF.concat (TF.scalar 0) [ v, v' ]
        TF.gradients y [v,v'] >>= TF.run
    V.fromList [1 :: Float] @=? dv
    V.fromList [1 :: Float] @=? dv'
    [dw,dw'] <- TF.runSession $ do
        w  <- TF.render $ TF.vector [1,2,3,4 :: Float]
        w' <- TF.render $ TF.vector [5,6,7,8 :: Float]
        let y = TF.concat (TF.scalar 0) [ w, w', w ]
        TF.gradients y [w,w'] >>= TF.run
    V.fromList [2,2,2,2 :: Float] @=? dw
    V.fromList [1,1,1,1 :: Float] @=? dw'

verifyConcatGradients :: [[Int64]] -> Int32  -> IO ()
verifyConcatGradients shapes concatDim = do
    let floatsFromShape :: [Int64] -> IO [Float]
        floatsFromShape shape = replicateM (fromIntegral $ List.product shape) randomIO
        constantZip = zipWithM $ \x shape -> TF.render $ TF.constant (TF.Shape shape) x
    inputGrads <- mapM floatsFromShape shapes
    inputs     <- mapM floatsFromShape shapes
    dinputs <- TF.runSession $ do
        inputTensors     <- inputs `constantZip` shapes
        inputGradTensors <- inputGrads `constantZip` shapes
        inputTensor      <- TF.render $ TF.concat (TF.scalar concatDim) inputTensors
        inputGradTensor  <- TF.render $ TF.concat (TF.scalar concatDim) inputGradTensors
        output           <- TF.render $ inputTensor `TF.mul` inputGradTensor
        TF.gradients output inputTensors >>= TF.run
    (V.fromList <$> inputGrads) @=? dinputs

-- This test checks that the gradient of a concat op
--   is correct along the first, second, and third dimension.
testConcatGradientSimple :: Test
testConcatGradientSimple = testCase "testConcatGradientSimple" $ do
    --   The following check is equivalent to ConcatTest._testGradientsSimple from
    --   tensorflow/tensorflow/compiler/tests/concat_ops_test.py
    verifyConcatGradients [[10,x,2] | x <- [1,2,6]] 1
    --   The following check is equivalent to ConcatTest._testGradientsFirstDim from
    --   tensorflow/tensorflow/compiler/tests/concat_ops_test.py
    verifyConcatGradients [[x,10,2] | x <- [1,2,6]] 0
    --   The following check is equivalent to ConcatTest._testGradientsLastDim from
    --   tensorflow/tensorflow/compiler/tests/concat_ops_test.py
    verifyConcatGradients [[10,2,x] | x <- [1,2,6]] 2


-- This test checks that the gradient of a concat op
--   along a random dimension across random shapes is as expected.
--   This test is inspired by ConcatTest._RunAndVerifyGradientsRandom from
--   tensorflow/tensorflow/compiler/tests/concat_ops_test.py, but also
--   verifies the gradient along negative concat dimensions.
testConcatRunAndVerifyGradientsRandom :: Test
testConcatRunAndVerifyGradientsRandom = testCase "testConcatRunAndVerifyGradientsRandom" $
    forM_ [1..5 :: Int] $ \_ -> do
        (shapes' :: [Int64]) <- replicateM 5 $ randomRIO (1, 5)
        (numTensors :: Int)  <- randomRIO (2, 10)
        (concatDim :: Int)   <- randomRIO (-4, 4)
        (concatDimSizes :: [Int64]) <- replicateM numTensors $ randomRIO (1, 5)
        let update i xs x = take i xs ++ x: drop (i+1) xs
            concatDim'    = concatDim `mod` length shapes'
            shapes        = map (update concatDim' shapes') concatDimSizes
        verifyConcatGradients shapes $ fromIntegral concatDim

-- run single test like this:
-- stack --docker --docker-image=$IMAGE_NAME test tensorflow-ops:GradientTest --test-arguments -t"*MaximumGrad*"
testMaximumGrad :: Test
testMaximumGrad = testCase "testMaximumGrad" $ do
    [gx, gy] <- TF.runSession $ do
        x <- TF.render $ TF.vector [0 :: Float]
        y <- TF.render $ TF.vector [0 :: Float]
        let z = TF.maximum x y
        TF.gradients z [x, y] >>= TF.run
    V.fromList [1] @=? gx
    V.fromList [1] @=? gy

testMaximumGradGrad :: Test
testMaximumGradGrad = testCase "testMaximumGradGrad" $ do
    [ggx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [2 :: Float]
        y <- TF.render $ TF.vector [1 :: Float]
        let z = TF.maximum x y
        [gx, _gy] <- TF.gradients z [x, y]
        TF.gradients gx [x] >>= TF.run
    V.fromList [0] @=? ggx

testReluGrad :: Test
testReluGrad = testCase "testReluGrad" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [2 :: Float]
        let y = TF.relu x
        TF.gradients y [x] >>= TF.run
    V.fromList [1] @=? dx

testReluGradGrad :: Test
testReluGradGrad = testCase "testReluGradGrad" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [2 :: Float]
        let y = TF.relu x
        [y'] <- TF.gradients y [x]
        TF.gradients y' [x] >>= TF.run
    V.fromList [0] @=? dx

testFillGrad :: Test
testFillGrad = testCase "testFillGrad" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.scalar (9 :: Float)
        let shape = TF.vector [2, 3 :: Int32]
        let y = TF.fill shape x
        TF.gradients y [x] >>= TF.run
    V.fromList [6] @=? dx


testTileGrad :: Test
testTileGrad = testCase "testTileGrad" $ do
    [dx] <- TF.runSession $ do
        x <- TF.render $ TF.vector [5, 9 :: Float]
        let multiples = TF.vector [2 :: Int32]
        let y = TF.tile x multiples
        TF.gradients y [x] >>= TF.run
    V.fromList [2, 2] @=? dx


testTile2DGrad :: Test
testTile2DGrad = testCase "testTileGrad2D" $ do
    (dx, shapeDX, shapeX) <- TF.runSession $ do
        let shape = TF.vector [3, 2 :: Int32]
        x <- TF.render $ TF.fill shape (TF.scalar (1::Float))
        let multiples = TF.vector [2, 3 :: Int32]
        let y = TF.tile x multiples

        [dx] <- TF.gradients y [x]
        TF.run (dx, TF.shape dx, TF.shape x)
    shapeX @=? (shapeDX :: V.Vector Int32)
    V.fromList [6, 6, 6, 6, 6, 6::Float] @=? (dx :: V.Vector Float)


matMulGradient :: Test
matMulGradient = testCase "matMulGradients" $ do

  let dfBuild = do
        x <- TF.render $ TF.zeros $ TF.Shape [3, 1 :: Int64]
        w <- TF.zeroInitializedVariable $ TF.Shape [1, 2 :: Int64]
        let f = x `TF.matMul` TF.readValue w :: TF.Tensor TF.Build Float
        dfs <- TF.gradients f [x]
        return (x, dfs)

  (xShape, dxShape) <- TF.runSession $ do
    (x, [dx]) <- TF.build dfBuild
    TF.run (TF.shape x, TF.shape dx)

  assertEqual "Shape of gradient must match shape of input" xShape (dxShape :: V.Vector Int32)


-- test that gradient of matMul can be taken gradient of
matMulGradGrad :: Test
matMulGradGrad = testCase "matMulGradGrad" $ do
  let width = 2 :: Int64
      batch = 4 :: Int64

  let tower = do
        x <- TF.render $ TF.zeros $ TF.Shape [batch, 1]
        w <- TF.zeroInitializedVariable $ TF.Shape [1, width]
        let f = x `TF.matMul` TF.readValue w
        [dfdx] <- TF.gradients f [x]
        let f'x = TF.reduceSum dfdx
        [dfdw] <- TF.gradients f'x [w] -- take gradient again (this time over w)
        return [TF.readValue w, TF.expr dfdw]

  TF.runSession $ do
    [w, dfdw] <- TF.build tower
    (wShape, dfdwShape) <- TF.run (TF.shape w, TF.shape dfdw)
    liftIO $ assertEqual "Shape of gradient must match input" wShape (dfdwShape :: V.Vector Int32)

    let step = w `TF.add` dfdw
    w0 <- TF.run step
    liftIO $ V.fromList [4, 4 :: Float] @=? w0


-- test that gradient of matMul deals correctly with transpose_a and transpose_b
matMulTransposeGradient :: (Bool, Bool) -> Test
matMulTransposeGradient txw = testCase ("matMulTransposeGradients " ++ show txw) $ do
  let (transposeX, transposeW) = txw

  let dfBuild = do
        let xShape = TF.Shape [3, 1 :: Int64]
        let xZeros = TF.zeros xShape
        x <- TF.render $ if transposeX then TF.matTranspose xZeros else xZeros
        variable <- TF.zeroInitializedVariable $ TF.Shape [1, 2 :: Int64]
        let wv = if transposeW then TF.matTranspose (TF.readValue variable) else TF.readValue variable
        let f = TF.matMul' (transAttrs transposeX transposeW) x wv :: TF.Tensor TF.Build Float
        w <- TF.render wv
        ds <- TF.gradients f [x, w]
        return (x, w, ds)

  TF.runSession $ do
    (x, w, [dx, dw]) <- TF.build dfBuild
    xShape <- TF.run $ TF.shape x
    dxShape <- TF.run $ TF.shape dx
    liftIO $ assertEqual "xShape must match dxShape" xShape (dxShape :: V.Vector Int32)

    wShape <- TF.run $ TF.shape w
    dwShape <- TF.run $ TF.shape dw
    liftIO $ assertEqual "wShape must match dwShape" wShape (dwShape :: V.Vector Int32)

transAttrs :: (TF.Attribute a,
               TF.Attribute b) =>
              a -> b -> TF.OpDef -> TF.OpDef
transAttrs a b =
  (TF.opAttr "transpose_a" .~ a) . (TF.opAttr "transpose_b" .~ b)

main :: IO ()
main = defaultMain
            [ testGradientSimple
            , testGradientDisconnected
            , testCreateGraphStateful
            , testCreateGraphNameScopes
            , testDiamond
            , testAddNGradient
            , testMaxGradient
            , testConcatGradient
            , testConcatGradientSimple
            , testConcatRunAndVerifyGradientsRandom
            , testMaximumGrad
            , testMaximumGradGrad
            , testReluGrad
            , testReluGradGrad
            , testFillGrad
            , testTileGrad
            , testTile2DGrad
            , matMulGradient
            , matMulGradGrad
            , matMulTransposeGradient (False, False)
            , matMulTransposeGradient (False, True)
            , matMulTransposeGradient (True, False)
            , matMulTransposeGradient (True, True)
            ]
