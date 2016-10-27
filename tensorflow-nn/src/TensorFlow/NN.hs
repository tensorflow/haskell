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
{-# LANGUAGE OverloadedStrings #-}

module TensorFlow.NN
    ( sigmoidCrossEntropyWithLogits
    ) where

import Prelude hiding (log, exp)
import TensorFlow.Build
import TensorFlow.Tensor
import TensorFlow.Types
import TensorFlow.Ops
import TensorFlow.GenOps.Core (greaterEqual, select, log, exp)

sigmoidCrossEntropyWithLogits
  :: (OneOf '[Float, Double] a, TensorType a, Num a) =>
     Tensor Value a
     -> Tensor Value a -> Build (Tensor Value a)
sigmoidCrossEntropyWithLogits logits targets = do
    let zeros = zerosLike logits
        cond = logits `greaterEqual` zeros
        relu_logits = select cond logits zeros
        neg_abs_logits = select cond (-logits) logits
    withNameScope "logistic_loss" $ do
        left  <- render $ relu_logits - logits * targets
        right <- render $ log (1 + exp neg_abs_logits)
        withNameScope "sigmoid_add" $ render $ left `add` right
