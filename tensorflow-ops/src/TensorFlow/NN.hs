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

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module TensorFlow.NN
    ( sigmoidCrossEntropyWithLogits
    ) where

import Prelude hiding           ( log
                                , exp
                                )
import TensorFlow.Build         ( MonadBuild
                                , withNameScope
                                )
import TensorFlow.GenOps.Core   ( greaterEqual
                                , select
                                , log
                                , exp
                                )
import TensorFlow.Tensor        ( Tensor(..)
                                , render
                                , Value
                                )
import TensorFlow.Types         ( TensorType(..)
                                , OneOf
                                )
import TensorFlow.Ops           ( zerosLike
                                , add
                                , mul
                                , neg
                                )

-- | Computes sigmoid cross entropy given `logits`.
--
-- Measures the probability error in discrete classification tasks in which each
-- class is independent and not mutually exclusive.  For instance, one could
-- perform multilabel classification where a picture can contain both an elephant
-- and a dog at the same time.
--
-- For brevity, let `x = logits`, `z = targets`.  The logistic loss is
--
--        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
--      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
--      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
--      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
--      = (1 - z) * x + log(1 + exp(-x))
--      = x - x * z + log(1 + exp(-x))
--
--  For x < 0, to avoid overflow in exp(-x), we reformulate the above
--
--        x - x * z + log(1 + exp(-x))
--      = log(exp(x)) - x * z + log(1 + exp(-x))
--      = - x * z + log(1 + exp(x))
--
--  Hence, to ensure stability and avoid overflow, the implementation uses this
--  equivalent formulation
--
--      max(x, 0) - x * z + log(1 + exp(-abs(x)))
--
--  `logits` and `targets` must have the same type and shape.
sigmoidCrossEntropyWithLogits
  :: (MonadBuild m, OneOf '[Float, Double] a, TensorType a, Num a)
     => Tensor Value a          -- ^ __logits__
     -> Tensor Value a          -- ^ __targets__
     -> m (Tensor Value a)
sigmoidCrossEntropyWithLogits logits targets = do
    let zeros = zerosLike logits
        cond = logits `greaterEqual` zeros
        relu_logits = select cond logits zeros
        neg_abs_logits = select cond (neg logits) logits
    withNameScope "logistic_loss" $ do
        left <- render $ relu_logits - logits `mul` targets
        right <- render $ log (1 + exp neg_abs_logits)
        withNameScope "sigmoid_add" $ render $ left `add` right
