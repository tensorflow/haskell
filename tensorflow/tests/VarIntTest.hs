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

module Main where

import Data.ByteString.Builder (toLazyByteString)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.QuickCheck2 (testProperty)
import qualified Data.Attoparsec.ByteString.Lazy as Atto

import TensorFlow.Internal.VarInt

testEncodeDecode :: Test
testEncodeDecode = testProperty "testEncodeDecode" $ \x ->
    let bytes = toLazyByteString (putVarInt x)
    in case Atto.eitherResult $ Atto.parse getVarInt bytes of
        Left _ -> False
        Right y -> x == y

main :: IO ()
main = defaultMain [ testEncodeDecode
                   ]
