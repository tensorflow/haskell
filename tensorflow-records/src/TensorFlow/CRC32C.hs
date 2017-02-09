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

-- | FFI wrappers of CRC32C digest.  Import qualified.

module TensorFlow.CRC32C
  ( value, extend
  , mask, unmask
  , valueMasked
  ) where

import Data.Bits (rotateL, rotateR)
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import Data.List (foldl')
import Data.Word (Word32)

-- | Compute the CRC32C checksum of the concatenation of the bytes checksummed
-- by the given CRC32C value and the bytes in the given ByteString.
extend :: Word32 -> B.ByteString -> Word32
extend = _crcExtend

-- | Compute the CRC32C checksum of the given bytes.
value :: BL.ByteString -> Word32
value = foldl' extend 0 . BL.toChunks

-- | Scramble a CRC32C value so that the result can be safely stored in a
-- bytestream that may itself be CRC'd.
--
-- This masking is the algorithm specified by TensorFlow's TFRecords format.
mask :: Word32 -> Word32
mask x = rotateR x 15 + maskDelta

-- | Inverse of 'mask'.
unmask :: Word32 -> Word32
unmask x = rotateL (x - maskDelta) 15

-- | Convenience function combining 'value' and 'mask'.
valueMasked :: BL.ByteString -> Word32
valueMasked = mask . value

maskDelta :: Word32
maskDelta = 0xa282ead8
