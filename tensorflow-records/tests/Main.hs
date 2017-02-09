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
module Main where

import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import Data.Monoid ((<>))
import Data.Word (Word8)
import Data.Serialize (runGet, runPut)
import Test.Framework (Test, defaultMain)
import Test.Framework.Providers.QuickCheck2 (testProperty)

import TensorFlow.Records (getTFRecord, putTFRecord)

main :: IO ()
main = defaultMain tests

tests :: [Test]
tests =
    [ testProperty "Inverse" propEncodeDecodeInverse
    , testProperty "FixedRecord" propFixedRecord
    ]

-- There's no (Arbitrary BL.ByteString), so pack it from a list of chunks.
propEncodeDecodeInverse :: [[Word8]] -> Bool
propEncodeDecodeInverse s =
    let bs = BL.fromChunks . fmap B.pack $ s
    in  runGet getTFRecord (runPut (putTFRecord bs)) == Right bs

propFixedRecord :: Bool
propFixedRecord =
    ("\x42" == case runGet getTFRecord record of
        Left err -> error err  -- Make the error appear in the test failure.
        Right x -> x) &&
    (runPut (putTFRecord "\x42") == record)
  where
    record = "\x01\x00\x00\x00\x00\x00\x00\x00" <> "\x01\x75\xde\x41" <>
             "\x42" <> "\x52\xcf\xb8\x1e"
