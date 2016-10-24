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

{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

-- | Parallel lookups on the list of tensors.
module TensorFlow.EmbeddingOps where

import Control.Monad (zipWithM)
import Data.Int (Int32, Int64)
import Data.List (genericLength)
import TensorFlow.Build (Build, colocateWith, render)
import TensorFlow.Ops ()  -- Num instance for Tensor
import TensorFlow.Tensor (Tensor, Value)
import TensorFlow.Types (OneOf, TensorType)
import qualified TensorFlow.GenOps.Core as CoreOps

-- | Looks up `ids` in a list of embedding tensors.
--
-- This function is used to perform parallel lookups on the list of
-- tensors in `params`.  It is a generalization of `TF.gather`, where
-- `params` is interpreted as a partition of a larger embedding
-- tensor.
--
-- The partition_strategy is "mod", we assign each id to partition
-- `p = id % len(params)`. For instance,
-- 13 ids are split across 5 partitions as:
-- `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`
--
-- The results of the lookup are concatenated into a dense
-- tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.
embeddingLookup :: forall a b v .
                   ( TensorType a
                   , OneOf '[Int64, Int32] b
                   , Num b
                   )
                => [Tensor v a]
                -- ^ A list of tensors which can be concatenated along
                -- dimension 0. Each `Tensor` must be appropriately
                -- sized for `mod` partition strategy.
                -> Tensor Value b
                -- ^ A `Tensor` with type `int32` or `int64`
                -- containing the ids to be looked up in `params`.
                -- The ids are required to be flat on entry and have
                -- fewer than 2^31 entries.
                -> Build (Tensor Value a)
                -- ^ A dense tensor with shape `shape(ids) + shape(params)[1:]`.
embeddingLookup params ids =
    CoreOps.dynamicStitch pindices <$> partitionedResult
  where np = genericLength params
        pAssignments = CoreOps.cast (ids `CoreOps.mod` np)
        newIds = ids `CoreOps.div` np
        originalIndices = CoreOps.range 0 (CoreOps.size ids) 1
        -- Partition list of ids based on assignments into np separate lists
        gatherIds = CoreOps.dynamicPartition np newIds pAssignments
        -- Similarly, partition the original indices.
        pindices = CoreOps.dynamicPartition np originalIndices pAssignments
        -- Do np separate lookups, finding embeddings for plist[p] in params[p]
        partitionedResult = zipWithM
                            (\p g -> colocateWith p $ render $ CoreOps.gather p g)
                            params gatherIds
