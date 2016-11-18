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

module TensorFlow.Test
    ( assertAllClose
    ) where

import qualified Data.Vector as V
import Test.HUnit ((@?))
import Test.HUnit.Lang (Assertion)
-- | Compares that the vectors are element-by-element equal within the given
-- tolerance. Raises an assertion and prints some information if not.
assertAllClose :: V.Vector Float -> V.Vector Float -> Assertion
assertAllClose xs ys = all (<= tol) (V.zipWith absDiff xs ys) @?
    "Difference > tolerance: \nxs: " ++ show xs ++ "\nys: " ++ show ys
        ++ "\ntolerance: " ++ show tol
  where
      absDiff x y = abs (x - y)
      tol = 0.001 :: Float
