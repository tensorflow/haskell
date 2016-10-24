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

{-# LANGUAGE BangPatterns #-}

{-|
Module      : TensorFlow.Internal.VarInt
Description : Encoders and decoders for varint types.

Originally taken from internal proto-lens code.
-}
module TensorFlow.Internal.VarInt
    ( getVarInt
    , putVarInt
    ) where

import Data.Attoparsec.ByteString as Parse
import Data.Bits
import Data.ByteString.Lazy.Builder as Builder
import Data.Monoid ((<>))
import Data.Word (Word64)

-- | Decode an unsigned varint.
getVarInt :: Parser Word64
getVarInt = loop 1 0
  where
    loop !s !n = do
        b <- anyWord8
        let n' = n + s * fromIntegral (b .&. 127)
        if (b .&. 128) == 0
            then return n'
            else loop (128*s) n'

-- | Encode a Word64.
putVarInt :: Word64 -> Builder
putVarInt n
    | n < 128 = Builder.word8 (fromIntegral n)
    | otherwise = Builder.word8 (fromIntegral $ n .&. 127 .|. 128)
                      <> putVarInt (n `shiftR` 7)
