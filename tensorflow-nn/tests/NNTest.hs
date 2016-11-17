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
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import           Data.Maybe                         (fromMaybe)
import           Google.Test                        (googleTest)
import           TensorFlow.Test                    (assertAllClose)
import           Test.Framework.Providers.HUnit     (testCase)
import           Test.HUnit                         ((@?))
import           Test.HUnit.Lang                    (Assertion(..))
import qualified Data.Vector                        as V
import qualified TensorFlow.Build                   as TF
import qualified TensorFlow.Gradient                as TF
import qualified TensorFlow.NN                      as TF
import qualified TensorFlow.Ops                     as TF
import qualified TensorFlow.Session                 as TF
import qualified TensorFlow.Tensor                  as TF
import qualified TensorFlow.Types                   as TF

-- | These tests are ported from:
--
--      <tensorflow>/tensorflow/python/ops/nn_xent_tests.py
--
-- This is the implementation we use to check the implementation we
-- wrote in `TensorFlow.NN.sigmoidCrossEntropyWithLogits`.
--
sigmoidXentWithLogits :: Floating a => Ord a => [a] -> [a] -> [a]
sigmoidXentWithLogits logits' targets' =
    let sig  = map (\x -> 1 / (1 + exp (-x))) logits'
        eps  = 0.0001
        pred = map (\p -> min (max p eps) (1 - eps)) sig
        xent y z = (-z) * (log y) - (1 - z) * log (1 - y)
     in zipWith xent pred targets'


data Inputs = Inputs {
      logits  :: [Float]
    , targets :: [Float]
    }


defInputs :: Inputs
defInputs = Inputs {
      logits    = [-100, -2, -2, 0, 2, 2,   2, 100]
    , targets   = [   0,  0,  1, 0, 0, 1, 0.5,   1]
    }


testLogisticOutput = testCase "testLogisticOutput" $ do
    let inputs     = defInputs
        vLogits    = TF.vector $ logits  inputs
        vTargets   = TF.vector $ targets inputs
        tfLoss     = TF.sigmoidCrossEntropyWithLogits vLogits vTargets
        ourLoss    = V.fromList $ sigmoidXentWithLogits (logits inputs) (targets inputs)

    r <- run tfLoss
    assertAllClose r ourLoss


testLogisticOutputMultipleDim =
        testCase "testLogisticOutputMultipleDim" $ do
    let inputs   = defInputs
        shape    = [2, 2, 2]
        vLogits  = TF.constant shape (logits  inputs)
        vTargets = TF.constant shape (targets inputs)
        tfLoss   = TF.sigmoidCrossEntropyWithLogits vLogits vTargets
        ourLoss  = V.fromList $ sigmoidXentWithLogits (logits inputs) (targets inputs)

    r <- run tfLoss
    assertAllClose r ourLoss


testGradientAtZero = testCase "testGradientAtZero" $ do
    let inputs   = defInputs { logits = [0, 0], targets = [0, 1] }
        vLogits  = TF.vector $ logits  inputs
        vTargets = TF.vector $ targets inputs
        tfLoss   = TF.sigmoidCrossEntropyWithLogits vLogits vTargets

    r <- run $ do
        l <- tfLoss
        TF.gradients l [vLogits]

    assertAllClose (head r) (V.fromList [0.5, -0.5])


run = TF.runSession . TF.buildAnd TF.run


main :: IO ()
main = googleTest [ testGradientAtZero
                  , testLogisticOutput
                  , testLogisticOutputMultipleDim
                  ]
