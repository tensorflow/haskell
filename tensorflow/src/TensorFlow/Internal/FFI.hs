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

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Internal.FFI
    ( TensorFlowException(..)
    , Raw.Session
    , withSession
    , extendGraph
    , run
    , TensorData(..)
    , setSessionConfig
    , setSessionTarget
    , getAllOpList
      -- * Internal helper.
    , useProtoAsVoidPtrLen
    )
    where

import Control.Concurrent.Async (Async, async, cancel, waitCatch)
import Control.Concurrent.MVar (MVar, modifyMVarMasked_, newMVar, takeMVar)
import Control.Exception (Exception, throwIO, bracket, finally, mask_)
import Control.Monad (when)
import Data.Bits (Bits, toIntegralSized)
import Data.Int (Int64)
import Data.Maybe (fromMaybe)
import Data.Typeable (Typeable)
import Data.Word (Word8)
import Foreign (Ptr, FunPtr, nullPtr, castPtr)
import Foreign.C.String (CString)
import Foreign.ForeignPtr (newForeignPtr, newForeignPtr_, withForeignPtr)
import Foreign.Marshal.Alloc (free)
import Foreign.Marshal.Array (withArrayLen, peekArray, mallocArray, copyArray)
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString as B
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.Encoding.Error as T
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import Data.ProtoLens (Message, encodeMessage)
import Proto.Tensorflow.Core.Framework.Graph (GraphDef)
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import Proto.Tensorflow.Core.Protobuf.Config (ConfigProto)

import qualified TensorFlow.Internal.Raw as Raw

data TensorFlowException = TensorFlowException Raw.Code T.Text
    deriving (Show, Eq, Typeable)

instance Exception TensorFlowException

-- | All of the data needed to represent a tensor.
data TensorData = TensorData
    { tensorDataDimensions :: [Int64]
    , tensorDataType       :: !DataType
    , tensorDataBytes      :: !(S.Vector Word8)
    }
  deriving (Show, Eq)

-- | Runs the given action after creating a session with options
-- populated by the given optionSetter.
withSession :: (Raw.SessionOptions -> IO ())
            -> ((IO () -> IO ()) -> Raw.Session -> IO a)
            -- ^ The action can spawn concurrent tasks which will
            -- be canceled before withSession returns.
            -> IO a
withSession optionSetter action = do
    drain <- newMVar []
    let cleanup s =
        -- Closes the session to nudge the pending run calls to fail and exit.
            finally (checkStatus (Raw.closeSession s)) $ do
                runners <- takeMVar drain
                -- Collects all runners before deleting the session.
                mapM_ shutDownRunner runners
                checkStatus (Raw.deleteSession s)
    bracket Raw.newSessionOptions Raw.deleteSessionOptions $ \options -> do
        optionSetter options
        bracket
            (checkStatus (Raw.newSession options))
            cleanup
            (action (asyncCollector drain))

asyncCollector :: MVar [Async ()] -> IO () -> IO ()
asyncCollector drain runner = modifyMVarMasked_ drain launchAndRecord
    where
      launchAndRecord restRunners = (: restRunners) <$> async runner

shutDownRunner :: Async () -> IO ()
shutDownRunner r = do
    cancel r
    -- TODO(gnezdo): manage exceptions better than print.
    either print (const (return ())) =<< waitCatch r

extendGraph :: Raw.Session -> GraphDef -> IO ()
extendGraph session pb =
    useProtoAsVoidPtrLen pb $ \ptr len ->
        checkStatus $ Raw.extendGraph session ptr len


run :: Raw.Session
    -> [(B.ByteString, TensorData)] -- ^ Feeds.
    -> [B.ByteString]               -- ^ Fetches.
    -> [B.ByteString]               -- ^ Targets.
    -> IO [TensorData]
run session feeds fetches targets = do
    let nullTensor = Raw.Tensor nullPtr
    -- Use mask to avoid leaking input tensors before they are passed to 'run'
    -- and output tensors before they are passed to 'createTensorData'.
    mask_ $
        -- Feeds
        withStringArrayLen (fst <$> feeds) $ \feedsLen feedNames ->
        mapM (createRawTensor . snd) feeds >>= \feedTensors ->
        withArrayLen feedTensors $ \_ cFeedTensors ->
        -- Fetches.
        withStringArrayLen fetches $ \fetchesLen fetchNames ->
        -- tensorOuts is an array of null Tensor pointers that will be filled
        -- by the call to Raw.run.
        withArrayLen (replicate fetchesLen nullTensor) $ \_ tensorOuts ->
        -- Targets.
        withStringArrayLen targets $ \targetsLen ctargets -> do
            checkStatus $ Raw.run
                session
                nullPtr
                feedNames cFeedTensors (safeConvert feedsLen)
                fetchNames tensorOuts (safeConvert fetchesLen)
                ctargets (safeConvert targetsLen)
                nullPtr
            mapM_ Raw.deleteTensor feedTensors
            outTensors <- peekArray fetchesLen tensorOuts
            mapM createTensorData outTensors


-- Internal.


-- | Same as 'fromIntegral', but throws an error if conversion is "lossy".
safeConvert ::
    forall a b. (Show a, Show b, Bits a, Bits b, Integral a, Integral b)
    => a -> b
safeConvert x =
    fromMaybe
    (error ("Failed to convert " ++ show x ++ ", got " ++
            show (fromIntegral x :: b)))
    (toIntegralSized x)


-- | Use a list of ByteString as a list of CString.
withStringList :: [B.ByteString] -> ([CString] -> IO a) -> IO a
withStringList strings fn = go strings []
  where
    go [] cs = fn (reverse cs)
    -- TODO(fmayle): Is it worth using unsafeAsCString here?
    go (x:xs) cs = B.useAsCString x $ \c -> go xs (c:cs)


-- | Use a list of ByteString as an array of CString.
withStringArrayLen :: [B.ByteString] -> (Int -> Ptr CString -> IO a) -> IO a
withStringArrayLen xs fn = withStringList xs (`withArrayLen` fn)


-- | Create a Raw.Tensor from a TensorData.
createRawTensor :: TensorData -> IO Raw.Tensor
createRawTensor (TensorData dims dt byteVec) =
    withArrayLen (map safeConvert dims) $ \cdimsLen cdims -> do
        let len = S.length byteVec
        dest <- mallocArray len
        S.unsafeWith byteVec $ \x -> copyArray dest x len
        Raw.newTensor (toEnum $ fromEnum dt)
                      cdims (safeConvert cdimsLen)
                      (castPtr dest) (safeConvert len)
                      tensorDeallocFunPtr nullPtr

{-# NOINLINE tensorDeallocFunPtr #-}
tensorDeallocFunPtr :: FunPtr Raw.TensorDeallocFn
tensorDeallocFunPtr = unsafePerformIO $ Raw.wrapTensorDealloc $ \x _ _ -> free x

-- | Create a TensorData from a Raw.Tensor.
--
-- Takes ownership of the Raw.Tensor.
-- TODO: Currently, it just makes a copy of the Tensor (and then deletes it),
-- since the raw pointer may refer to storage inside a mutable TensorFlow
-- variable.  We should avoid that copy when it's not needed; for example,
-- by making TensorData wrap an IOVector, and changing the code that uses it.
createTensorData :: Raw.Tensor -> IO TensorData
createTensorData t = do
    -- Read dimensions.
    numDims <- Raw.numDims t
    dims <- mapM (Raw.dim t) [0..numDims-1]
    -- Read type.
    dtype <- toEnum . fromEnum <$> Raw.tensorType t
    -- Read data.
    len <- safeConvert <$> Raw.tensorByteSize t
    bytes <- castPtr <$> Raw.tensorData t :: IO (Ptr Word8)
    fp <- newForeignPtr_ bytes
    -- Make an explicit copy of the raw data, since it might point
    -- to a mutable variable's memory.
    v <- S.freeze (M.unsafeFromForeignPtr0 fp len)
    Raw.deleteTensor t
    return $ TensorData (map safeConvert dims) dtype v

-- | Runs the given action which does FFI calls updating a provided
-- status object. If the status is not OK it is thrown as
-- TensorFlowException.
checkStatus :: (Raw.Status -> IO a) -> IO a
checkStatus fn =
    bracket Raw.newStatus Raw.deleteStatus $ \status -> do
        result <- fn status
        code <- Raw.getCode status
        when (code /= Raw.TF_OK) $ do
            msg <- T.decodeUtf8With T.lenientDecode <$>
                   (Raw.message status >>= B.packCString)
            throwIO $ TensorFlowException code msg
        return result

setSessionConfig :: ConfigProto -> Raw.SessionOptions -> IO ()
setSessionConfig pb opt =
    useProtoAsVoidPtrLen pb $ \ptr len ->
        checkStatus (Raw.setConfig opt ptr len)

setSessionTarget :: B.ByteString -> Raw.SessionOptions -> IO ()
setSessionTarget target = B.useAsCString target . Raw.setTarget

-- | Serializes the given msg and provides it as (ptr,len) argument
-- to the given action.
useProtoAsVoidPtrLen :: (Message msg, Integral c, Show c, Bits c) =>
                        msg -> (Ptr b -> c -> IO a) -> IO a
useProtoAsVoidPtrLen msg f = B.useAsCStringLen (encodeMessage msg) $
        \(bytes, len) -> f (castPtr bytes) (safeConvert len)

-- | Returns the serialized OpList of all OpDefs defined in this
-- address space.
getAllOpList :: IO B.ByteString
getAllOpList = do
    foreignPtr <-
        mask_ (newForeignPtr Raw.deleteBuffer =<< checkCall)
    -- Makes a copy because it is more reliable than eviscerating
    -- Buffer to steal its memory (including custom deallocator).
    withForeignPtr foreignPtr $
        \ptr -> B.packCStringLen =<< (,)
                <$> (castPtr <$> Raw.getBufferData ptr)
                <*> (safeConvert <$> Raw.getBufferLength ptr)
    where
      checkCall = do
          p <- Raw.getAllOpList
          when (p == nullPtr) (throwIO exception)
          return p
      exception = TensorFlowException
                Raw.TF_UNKNOWN "GetAllOpList failure, check logs"
