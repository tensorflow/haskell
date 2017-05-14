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
{-# LANGUAGE OverloadedLists #-}

module Main where

import           TensorFlow.Test                    (assertAllClose)
import           Test.Framework (defaultMain, Test)
import           Test.Framework.Providers.HUnit     (testCase)
import qualified Data.Vector                        as V
import qualified TensorFlow.Gradient                as TF
import qualified TensorFlow.NN                      as TF
import qualified TensorFlow.Ops                     as TF
import qualified TensorFlow.Core                    as TF

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
        predictions = map (\p -> min (max p eps) (1 - eps)) sig
        xent y z = (-z) * (log y) - (1 - z) * log (1 - y)
     in zipWith xent predictions targets'


data Inputs = Inputs {
      logits  :: [Float]
    , targets :: [Float]
    }


defInputs :: Inputs
defInputs = Inputs {
      logits    = [-100, -2, -2, 0, 2, 2,   2, 100]
    , targets   = [   0,  0,  1, 0, 0, 1, 0.5,   1]
    }


testLogisticOutput :: Test
testLogisticOutput = testCase "testLogisticOutput" $ do
    let inputs     = defInputs
    r <- run $ do
        vLogits    <- TF.render $ TF.vector $ logits  inputs
        vTargets   <- TF.render $ TF.vector $ targets inputs
        TF.sigmoidCrossEntropyWithLogits vLogits vTargets
    let ourLoss    = V.fromList $ sigmoidXentWithLogits (logits inputs) (targets inputs)
    assertAllClose r ourLoss


testLogisticOutputMultipleDim :: Test
testLogisticOutputMultipleDim =
        testCase "testLogisticOutputMultipleDim" $ do
    let inputs   = defInputs
        shape    = [2, 2, 2]
    r <- run $ do
        vLogits  <- TF.render $ TF.constant shape (logits  inputs)
        vTargets <- TF.render $ TF.constant shape (targets inputs)
        TF.sigmoidCrossEntropyWithLogits vLogits vTargets
    let ourLoss  = V.fromList $ sigmoidXentWithLogits (logits inputs) (targets inputs)
    assertAllClose r ourLoss


testGradientAtZero :: Test
testGradientAtZero = testCase "testGradientAtZero" $ do
    r <- run $ do
        let inputs   = defInputs { logits = [0, 0], targets = [0, 1] }
        vTargets <- TF.render $ TF.vector $ targets inputs
        vLogits  <- TF.render $ TF.vector $ logits  inputs
        let tfLoss   = TF.sigmoidCrossEntropyWithLogits vLogits vTargets

        l <- tfLoss
        TF.gradients l [vLogits]

    assertAllClose (head r) (V.fromList [0.5, -0.5])

run :: TF.Fetchable t a => TF.Session t -> IO a
run = TF.runSession . (>>= TF.run)

main :: IO ()
main = defaultMain
            [ testGradientAtZero
            , testLogisticOutput
            , testLogisticOutputMultipleDim
            ]
