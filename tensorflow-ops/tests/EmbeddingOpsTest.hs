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

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Tests for EmbeddingOps.
module Main where

import Data.Int (Int32, Int64)
import Data.List (genericLength)
import Google.Test (googleTest)
import TensorFlow.EmbeddingOps (embeddingLookup)
import Test.Framework.Providers.QuickCheck2 (testProperty)
import Test.HUnit ((@=?))
import Test.QuickCheck (Arbitrary(..), Property, choose, vectorOf)
import Test.QuickCheck.Monadic (monadicIO, run)

import qualified Data.Vector as V
import qualified TensorFlow.GenOps.Core as CoreOps
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF

-- Verifies that direct gather is the same as dynamic split into
-- partitions, followed by embedding lookup.
testEmbeddingLookupUndoesSplit ::
    forall a. (TF.TensorType a, TF.TensorProtoLens a, Show a, Eq a)
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
        let maxDim = fromIntegral $ ceiling $ 100 ** (1 / fromIntegral rank)
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
       ]
