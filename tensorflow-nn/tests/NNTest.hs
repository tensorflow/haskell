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

module Main where

import Google.Test                      (googleTest)
import Test.Framework.Providers.HUnit   (testCase)
import Test.HUnit                       ((@?))

import qualified Data.Vector            as V
import qualified TensorFlow.Build       as TF
import qualified TensorFlow.Ops         as TF
import qualified TensorFlow.Session     as TF
import qualified TensorFlow.Tensor      as TF
import qualified TensorFlow.Types       as TF
import qualified TensorFlow.NN          as TF


sigmoidXentWithLogits :: Floating a => Ord a => [a] -> [a] -> [a]
sigmoidXentWithLogits logits' targets' =
    let pred  = map (\x -> 1 / (1 + exp (-x))) logits'
        eps   = 0.0001
        pred' = map (\p -> min (max p eps) (1 - eps)) pred
        f y z = (-z) * (log y) - (1 - z) * log (1 - y)
     in zipWith f pred' targets'


x, y :: [Float]
x = [-100, -2, -2, 0, 2, 2,   2, 100]
y = [   0,  0,  1, 0, 0, 1, 0.5,   1]
shape = TF.Shape [8]


logits, targets :: TF.Tensor TF.Value Float
logits  = TF.constant shape x
targets = TF.constant shape y


losses :: (TF.TensorType a, Floating a, Ord a) => [a] -> [a] -> [a]
losses x' y' = sigmoidXentWithLogits x' y'


testLogisticOutput = testCase "testLogisticOutput" $ do
    let loss    = TF.sigmoidCrossEntropyWithLogits logits targets
        ourLoss = V.fromList (losses x y)
    --
    r <- TF.runSession . TF.buildAnd TF.run $ loss
    (all id (V.zipWith (\a b -> abs (a - b) <= 0.001) r ourLoss)) @? ("Xents too different: \n" ++ (show r) ++ "\n" ++ (show ourLoss))


main :: IO ()
main = googleTest [ testLogisticOutput ]
