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

-- | Tests for FFI.

module Main where

import Data.ProtoLens (decodeMessage)
import Lens.Family2 (view)
import TensorFlow.Internal.FFI (getAllOpList)
import Test.HUnit (assertBool, assertFailure)
import Test.Framework (defaultMain)
import Test.Framework.Providers.HUnit (testCase)
import Proto.Tensorflow.Core.Framework.OpDef (OpList, op)

testParseAll :: IO ()
testParseAll = do
    opList <- getAllOpList
    either
        assertFailure
        (assertBool "Expected non-empty list of default Ops"
         . not . null . view op)
        (decodeMessage opList :: Either String OpList)

main :: IO ()
main = defaultMain
    [ testCase "ParseAllOps" testParseAll
    ]
