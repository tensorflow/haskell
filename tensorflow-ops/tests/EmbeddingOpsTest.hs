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

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Tests for EmbeddingOps.
module Main where

import Data.Int (Int32, Int64)
import Data.List (genericLength)
import Google.Test (googleTest)
import TensorFlow.EmbeddingOps (embeddingLookup)
import Test.Framework (Test)
import Test.Framework.Providers.QuickCheck2 (testProperty)
import Test.HUnit ((@=?))
import Test.Framework.Providers.HUnit (testCase)
import Test.QuickCheck (Arbitrary(..), Property, choose, vectorOf)
import Test.QuickCheck.Monadic (monadicIO, run)
import TensorFlow.Test (assertAllClose)

import qualified Data.Vector as V
import qualified TensorFlow.GenOps.Core as CoreOps
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Build as TF
import qualified TensorFlow.Nodes as TF


buildAndRun :: TF.Fetchable t a => TF.Build t -> IO a
buildAndRun = TF.runSession . TF.buildAnd TF.run


-- | Tries to perform a simple embedding lookup, with two partitions.
testEmbeddingLookupHasRightShapeWithPartition :: Test
testEmbeddingLookupHasRightShapeWithPartition =
        testCase "testEmbeddingLookupHasRightShapeWithPartition" $ do
    let embShape     = TF.Shape [1, 3] -- Consider a 3-dim embedding of two items.
    let embedding1  = [1, 1, 1 :: Int32]
    let embedding2  = [0, 0, 0 :: Int32]
    let embedding   = [ TF.constant embShape embedding1
                      , TF.constant embShape embedding2
                      ]

    let idValues  = [0, 1 :: Int32]
    let ids       = TF.constant (TF.Shape [1, 2]) idValues
    let op        = embeddingLookup embedding ids

    (values, shape) <- buildAndRun $ do
        vs <- op
        return (vs, TF.shape vs)

    -- This is the shape that is returned in the equiv. Python.
    shape  @=? V.fromList [1, 2, 3]

    -- "[0, 1]" should pull out the resulting vector.
    values @=? V.fromList [1, 1, 1, 0, 0, 0]


-- | Tries to perform a simple embedding lookup, with only a single partition.
testEmbeddingLookupHasRightShape :: Test
testEmbeddingLookupHasRightShape =
        testCase "testEmbeddingLookupHasRightShape" $ do
    -- Consider a 3-dim embedding of two items
    let embShape      = TF.Shape [2, 3]
    let embeddingInit = [ 1, 1, 1
                        , 0, 0, 0 :: Int32
                        ]

    let embedding = TF.constant embShape embeddingInit
    let idValues  = [0, 1 :: Int32]
    let ids       = TF.constant (TF.Shape [1, 2]) idValues
    let op        = embeddingLookup [embedding] ids

    (values, shape) <- buildAndRun $ do
        vs <- op
        return (vs, TF.shape vs)

    -- This is the shape that is returned in the equiv. Python.
    shape  @=? V.fromList [1, 2, 3]

    -- "[0, 1]" should pull out the resulting vector.
    values @=? V.fromList [1, 1, 1, 0, 0, 0]


-- | Check that we can calculate gradients w.r.t embeddings.
testEmbeddingLookupGradients :: Test
testEmbeddingLookupGradients = testCase "testEmbeddingLookupGradients" $ do
    -- Agrees with "embedding", so gradient should be zero.
    let xVals = V.fromList ([20, 20 :: Float])
    let shape = TF.Shape [2]

    gs <- TF.runSession $ do
        grads <- TF.build $ do
            let embShape      = TF.Shape [2, 1]
            let embeddingInit = [1, 20 ::Float]
            let idValues      = [1, 1 :: Int32]
            let ids           = TF.constant (TF.Shape [1, 2]) idValues

            x <- TF.placeholder (TF.Shape [2])
            embedding <- TF.initializedVariable
                            =<< TF.render (TF.constant embShape embeddingInit)

            op <- embeddingLookup [embedding] ids
            let twoNorm = CoreOps.square $ TF.abs (op - x)
                loss    = TF.mean twoNorm (TF.scalar (0 :: Int32))

            grad <- fmap head (TF.gradients loss [embedding])
            return $ \xs -> TF.runWithFeeds [TF.feed x xs] grad

        grads (TF.encodeTensorData shape xVals :: TF.TensorData Float)
    -- Gradients should be zero (or close)
    assertAllClose gs (V.fromList ([0, 0 :: Float]))


-- Verifies that direct gather is the same as dynamic split into
-- partitions, followed by embedding lookup.
testEmbeddingLookupUndoesSplit ::
    forall a. (TF.TensorDataType V.Vector a, Show a, Eq a)
    => LookupExample a -> Property
testEmbeddingLookupUndoesSplit
    (LookupExample numParts
                   shape@(TF.Shape (firstDim : restDims))
                   values
                   indices) =
    let modShardedValues :: [TF.Tensor TF.Value a] =
            CoreOps.dynamicPartition numParts shapedValues cyclicCounter
        cyclicCounter :: TF.Tensor TF.Value Int32 =
            TF.vector [0..fromIntegral firstDim-1]
            `CoreOps.mod` fromIntegral numParts
        indicesVector = TF.vector indices
        directs = CoreOps.gather shapedValues indicesVector
        shapedValues = TF.constant shape values
    in monadicIO $ run $ do
       (shapeOut, got, want :: V.Vector a) <-
           TF.runSession $ TF.buildAnd TF.run $ do
               embeddings <- embeddingLookup modShardedValues indicesVector
               return (TF.cast (TF.shape embeddings), embeddings, directs)
       -- Checks the explicitly documented invariant of embeddingLookup.
       shapeOut @=? V.fromList (genericLength indices : restDims)
       got @=? want
testEmbeddingLookupUndoesSplit _ = error "Bug in Arbitrary (LookupExample)"

-- | Consistent set of parameters for EmbeddingLookupUndoesSplit.
data LookupExample a = LookupExample
                       Int64  -- ^ number of ways to split.
                       TF.Shape  -- ^ shape of the generated tensor
                       [a]       -- ^ data for the tensor
                       [Int32]   -- ^ indices to split the tensor by
    deriving Show

instance Arbitrary a => Arbitrary (LookupExample a) where
    arbitrary = do
        rank <- choose (1, 4)
        -- Takes rank-th root of 100 to cap the tensor size.
        let maxDim = fromIntegral (ceiling doubleMaxDim :: Int64)
            doubleMaxDim :: Double
            doubleMaxDim = 100 ** (1 / fromIntegral rank)
        shape@(firstDim : _) <- vectorOf rank (choose (1, maxDim))
        values <- vectorOf (fromIntegral $ product shape) arbitrary
        numParts <- choose (2, 15)
        indSize <- choose (0, fromIntegral $ firstDim - 1)
        indices <- vectorOf indSize (choose (0, fromIntegral firstDim - 1))
        return $ LookupExample numParts (TF.Shape shape) values indices

main :: IO ()
main = googleTest
       [ testProperty "EmbeddingLookupUndoesSplit"
         (testEmbeddingLookupUndoesSplit :: LookupExample Double -> Property)
       , testEmbeddingLookupHasRightShape
       , testEmbeddingLookupHasRightShapeWithPartition
       , testEmbeddingLookupGradients
       ]
