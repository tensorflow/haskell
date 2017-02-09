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

-- | Encoder and decoder for the TensorFlow \"TFRecords\" format.

{-# LANGUAGE Rank2Types #-}
module TensorFlow.Records
  (
  -- * Records
    putTFRecord
  , getTFRecord
  , getTFRecords

  -- * Implementation

  -- | These may be useful for encoding or decoding to types other than
  -- 'ByteString' that have their own Cereal codecs.
  , getTFRecordLength
  , getTFRecordData
  , putTFRecordLength
  , putTFRecordData
  ) where

import Control.Exception (evaluate)
import Control.Monad (when)
import Data.ByteString.Unsafe (unsafePackCStringLen)
import qualified Data.ByteString.Builder as B (Builder)
import Data.ByteString.Builder.Extra (runBuilder, Next(..))
import qualified Data.ByteString.Lazy as BL
import Data.Serialize.Get
  ( Get
  , getBytes
  , getWord32le
  , getWord64le
  , getLazyByteString
  , isEmpty
  , lookAhead
  )
import Data.Serialize
  ( Put
  , execPut
  , putLazyByteString
  , putWord32le
  , putWord64le
  )
import Data.Word (Word8, Word64)
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Ptr (Ptr, castPtr)
import System.IO.Unsafe (unsafePerformIO)

import TensorFlow.CRC32C (crc32cLBSMasked, crc32cUpdate, crc32cMask)

-- | Parse one TFRecord.
getTFRecord :: Get BL.ByteString
getTFRecord = getTFRecordLength >>= getTFRecordData

-- | Parse many TFRecords as a list.  Note you probably want streaming instead
-- as provided by the tensorflow-records-conduit package.
getTFRecords :: Get [BL.ByteString]
getTFRecords = do
  e <- isEmpty
  if e then return [] else (:) <$> getTFRecord <*> getTFRecords

getCheckMaskedCRC32C :: BL.ByteString -> Get ()
getCheckMaskedCRC32C bs = do
  wireCRC <- getWord32le
  let maskedCRC = crc32cLBSMasked bs
  when (maskedCRC /= wireCRC) $ fail $
      "getCheckMaskedCRC32C: CRC mismatch, computed: " ++ show maskedCRC ++
      ", expected: " ++ show wireCRC

-- | Get a length and verify its checksum.
getTFRecordLength :: Get Word64
getTFRecordLength = do
  buf <- lookAhead (getBytes 8)
  getWord64le <* getCheckMaskedCRC32C (BL.fromStrict buf)

-- | Get a record payload and verify its checksum.
getTFRecordData :: Word64 -> Get BL.ByteString
getTFRecordData len = if len > 0x7fffffffffffffff
  then fail "getTFRecordData: Record size overflows Int64"
  else do
    bs <- getLazyByteString (fromIntegral len)
    getCheckMaskedCRC32C bs
    return bs

putMaskedCRC32C :: BL.ByteString -> Put
putMaskedCRC32C = putWord32le . crc32cLBSMasked

-- Runs a Builder that's known to write a fixed number of bytes on an 'alloca'
-- buffer, and runs the given IO action on the result.  Raises exceptions if
-- the Builder yields ByteString chunks or attempts to write more bytes than
-- expected.
unsafeWithFixedWidthBuilder :: Int -> B.Builder -> (Ptr Word8 -> IO r) -> IO r
unsafeWithFixedWidthBuilder n b act = allocaBytes n $ \ptr -> do
  (_, signal) <- runBuilder b ptr n
  case signal of
    Done -> act ptr
    More _ _ -> error "unsafeWithFixedWidthBuilder: Builder returned More."
    Chunk _ _ -> error "unsafeWithFixedWidthBuilder: Builder returned Chunk."

-- | Put a record length and its checksum.
putTFRecordLength :: Word64 -> Put
putTFRecordLength x =
  let put = putWord64le x
      len = 8
      crc = crc32cMask $ unsafePerformIO $
          -- Serialized Word64 is always 8 bytes, so we can go fast by using
          -- alloca.
          unsafeWithFixedWidthBuilder len (execPut put) $ \ptr -> do
              str <- unsafePackCStringLen (castPtr ptr, len)
              -- Force the result to ensure it's evaluated before freeing ptr.
              evaluate $ crc32cUpdate 0 str
  in  put *> putWord32le crc

-- | Put a record payload and its checksum.
putTFRecordData :: BL.ByteString -> Put
putTFRecordData bs = putLazyByteString bs *> putMaskedCRC32C bs

-- | Put one TFRecord with the given contents.
putTFRecord :: BL.ByteString -> Put
putTFRecord bs =
  putTFRecordLength (fromIntegral $ BL.length bs) *> putTFRecordData bs
