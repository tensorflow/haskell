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

module TensorFlow.CRC32C
  ( crc32c
  , crc32cLBS
  , crc32cUpdate
  , crc32cMasked
  , crc32cLBSMasked
  , crc32cMask
  , crc32cUnmask
  ) where

import Data.Bits (rotateL, rotateR)
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import Data.Digest.CRC32C (crc32c, crc32c_update)
import Data.List (foldl')
import Data.Word (Word32)

-- | Compute the CRC32C checksum of the concatenation of the bytes checksummed
-- by the given CRC32C value and the bytes in the given ByteString.
crc32cUpdate :: Word32 -> B.ByteString -> Word32
crc32cUpdate = crc32c_update

-- | Compute the CRC32C checksum of the given bytes.
crc32cLBS :: BL.ByteString -> Word32
crc32cLBS = foldl' crc32cUpdate 0 . BL.toChunks

-- | Scramble a CRC32C value so that the result can be safely stored in a
-- bytestream that may itself be CRC'd.
--
-- This masking is the algorithm specified by TensorFlow's TFRecords format.
crc32cMask :: Word32 -> Word32
crc32cMask x = rotateR x 15 + maskDelta

-- | Inverse of 'crc32cMask'.
crc32cUnmask :: Word32 -> Word32
crc32cUnmask x = rotateL (x - maskDelta) 15

-- | Convenience function combining 'crc32c' and 'crc32cMask'.
crc32cMasked :: B.ByteString -> Word32
crc32cMasked = crc32cMask . crc32c

-- | Convenience function combining 'crc32cLBS' and 'crc32cMask'.
crc32cLBSMasked :: BL.ByteString -> Word32
crc32cLBSMasked = crc32cMask . crc32cLBS

maskDelta :: Word32
maskDelta = 0xa282ead8
