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

{-# LANGUAGE ForeignFunctionInterface #-}

module TensorFlow.Internal.Raw where

#include "third_party/tensorflow/c/c_api.h"

import Data.ByteString (ByteString, packCString, useAsCString)
import Foreign
import Foreign.C

{# enum TF_DataType as DataType {} deriving (Show, Eq) #}
{# enum TF_Code as Code {} deriving (Show, Eq) #}


-- Status.
{# pointer *TF_Status as Status newtype #}

newStatus :: IO Status
newStatus = {# call TF_NewStatus as ^ #}

deleteStatus :: Status -> IO ()
deleteStatus = {# call TF_DeleteStatus as ^ #}

setStatus :: Status -> Code -> CString -> IO ()
setStatus s c = {# call TF_SetStatus as ^ #} s (fromIntegral $ fromEnum c)

getCode :: Status -> IO Code
getCode s = toEnum . fromIntegral <$> {# call TF_GetCode as ^ #} s

message :: Status -> IO CString
message = {# call TF_Message as ^ #}


-- TString.
{# pointer *TF_TString as TString newtype #}

sizeOfTString :: Int
sizeOfTString = 24

-- TF_TString_Type::TF_TSTR_OFFSET
tstringOffsetTypeTag :: Word32
tstringOffsetTypeTag = 2

stringGetDataPointer :: TString -> IO CString
stringGetDataPointer = {# call TF_StringGetDataPointer as ^ #}

stringGetSize :: TString -> IO CULong
stringGetSize = {# call TF_StringGetSize as ^ #}


-- Operation.
{# pointer *TF_Operation as Operation newtype #}
{# fun TF_OperationName as operationName { `Operation' } -> `ByteString' packCString* #}
{# fun TF_OperationNumOutputs as operationNumOutputs { `Operation' } -> `Int' #}

instance Storable Operation where
    sizeOf (Operation t) = sizeOf t
    alignment (Operation t) = alignment t
    peek p = fmap Operation (peek (castPtr p))
    poke p (Operation t) = poke (castPtr p) t


-- Output.
data Output = Output
    { outputOperation :: Operation
    , outputIndex     :: CInt
    }
{# pointer *TF_Output as OutputPtr -> Output #}

instance Storable Output where
    sizeOf _ = {# sizeof TF_Output #}
    alignment _ = {# alignof TF_Output #}
    peek p = Output <$> {# get TF_Output->oper #} p
                    <*> (fromIntegral <$> {# get TF_Output->index #} p)
    poke p (Output oper index) = do
        {# set TF_Output->oper #} p oper
        {# set TF_Output->index #} p $ fromIntegral index


-- Buffer.
data Buffer
{# pointer *TF_Buffer as BufferPtr -> Buffer #}

getBufferData :: BufferPtr -> IO (Ptr ())
getBufferData = {# get TF_Buffer->data #}

getBufferLength :: BufferPtr -> IO CULong
getBufferLength = {# get TF_Buffer->length #}

newBufferFromString :: Ptr () -> CULong -> IO BufferPtr
newBufferFromString = {# call TF_NewBufferFromString as ^ #}

deleteBuffer :: BufferPtr -> IO ()
deleteBuffer = {# call TF_DeleteBuffer as ^ #}

-- Tensor.
{# pointer *TF_Tensor as Tensor newtype #}

instance Storable Tensor where
    sizeOf (Tensor t) = sizeOf t
    alignment (Tensor t) = alignment t
    peek p = fmap Tensor (peek (castPtr p))
    poke p (Tensor t) = poke (castPtr p) t

-- A synonym for the int64_t type, which is used in the TensorFlow API.
-- On some platforms it's `long`; on others (e.g., Mac OS X) it's `long long`;
-- and as far as Haskell is concerned, those are distinct types (`CLong` vs
-- `CLLong`).
type CInt64 = {#type int64_t #}

{# pointer *size_t as CSizePtr -> CSize #}

newTensor :: DataType
          -> Ptr CInt64   -- dimensions array
          -> CInt         -- num dimensions
          -> Ptr ()       -- data
          -> CULong       -- data len
          -> FunPtr (Ptr () -> CULong -> Ptr () -> IO ())  -- deallocator
          -> Ptr ()       -- deallocator arg
          -> IO Tensor
newTensor dt = {# call TF_NewTensor as ^ #} (fromIntegral $ fromEnum dt)

deleteTensor :: Tensor -> IO ()
deleteTensor = {# call TF_DeleteTensor as ^ #}

tensorType :: Tensor -> IO DataType
tensorType t = toEnum . fromIntegral <$> {# call TF_TensorType as ^ #} t

numDims :: Tensor -> IO CInt
numDims = {# call TF_NumDims as ^ #}

dim :: Tensor -> CInt -> IO CInt64
dim = {# call TF_Dim as ^ #}

tensorByteSize :: Tensor -> IO CULong
tensorByteSize = {# call TF_TensorByteSize as ^ #}

tensorData :: Tensor -> IO (Ptr ())
tensorData = {# call TF_TensorData as ^ #}

-- ImportGraphDefOptions.
{# pointer *TF_ImportGraphDefOptions as ImportGraphDefOptions newtype #}
{# fun TF_NewImportGraphDefOptions as newImportGraphDefOptions { } -> `ImportGraphDefOptions' #}
{# fun TF_DeleteImportGraphDefOptions as deleteImportGraphDefOptions { `ImportGraphDefOptions' } -> `()' #}
{# fun TF_ImportGraphDefOptionsAddInputMapping as importGraphDefOptionsAddInputMapping
    {                `ImportGraphDefOptions'
    , useAsCString*  `ByteString'
    ,                `Int'
    ,               %`OutputPtr'
    } ->             `()'
#}


-- Graph.
{# pointer *TF_Graph as Graph newtype #}
{# fun TF_NewGraph as newGraph { } -> `Graph' #}
{# fun TF_DeleteGraph as deleteGraph { `Graph' } -> `()' #}
{# fun TF_GraphOperationByName as graphOperationByName
    {               `Graph'
    , useAsCString* `ByteString'
    } ->            `Operation'
#}
{# fun TF_GraphNextOperation as graphNextOperation { `Graph', `CSizePtr' } -> `Operation' #}
{# fun TF_GraphImportGraphDef as graphImportGraphDef { `Graph', `BufferPtr', `ImportGraphDefOptions', `Status' } -> `()' #}


-- Session Options.
{# pointer *TF_SessionOptions as SessionOptions newtype #}

newSessionOptions :: IO SessionOptions
newSessionOptions = {# call TF_NewSessionOptions as ^ #}

setTarget :: SessionOptions -> CString -> IO ()
setTarget = {# call TF_SetTarget as ^ #}

setConfig :: SessionOptions -> Ptr () -> CULong -> Status -> IO ()
setConfig = {# call TF_SetConfig as ^ #}

deleteSessionOptions :: SessionOptions -> IO ()
deleteSessionOptions = {# call TF_DeleteSessionOptions as ^ #}


-- Session.
{# pointer *TF_Session as Session newtype #}

newSession :: Graph -> SessionOptions -> Status -> IO Session
newSession = {# call TF_NewSession as ^ #}

{# fun TF_LoadSessionFromSavedModel as loadSessionFromSavedModel
    {                     `SessionOptions'
    ,                     `BufferPtr'     -- RunOptions proto.
    , useAsCString*       `ByteString'    -- Export directory.
    , withStringArrayLen* `[ByteString]'& -- Tags.
    ,                     `Graph'
    ,                     `BufferPtr'     -- MetaGraphDef.
    ,                     `Status'
    } ->                  `Session'
#}

closeSession :: Session -> Status -> IO ()
closeSession = {# call TF_CloseSession as ^ #}

deleteSession :: Session -> Status -> IO ()
deleteSession = {# call TF_DeleteSession as ^ #}

run :: Session
    -> BufferPtr                       -- RunOptions proto.
    -> OutputPtr -> Ptr Tensor -> CInt -- Input (names, tensors, count).
    -> OutputPtr -> Ptr Tensor -> CInt -- Output (names, tensors, count).
    -> Ptr Operation           -> CInt -- Target operations (ops, count).
    -> BufferPtr                       -- RunMetadata proto.
    -> Status
    -> IO ()
run = {# call TF_SessionRun as ^ #}

-- FFI helpers.
type TensorDeallocFn = Ptr () -> CULong -> Ptr () -> IO ()
foreign import ccall "wrapper"
    wrapTensorDealloc :: TensorDeallocFn -> IO (FunPtr TensorDeallocFn)


-- | Get the OpList of all OpDefs defined in this address space.
-- Returns a BufferPtr, ownership of which is transferred to the caller
-- (and can be freed using deleteBuffer).
--
-- The data in the buffer will be the serialized OpList proto for ops registered
-- in this address space.
getAllOpList :: IO BufferPtr
getAllOpList = {# call TF_GetAllOpList as ^ #}

-- | Use a list of ByteString as a list of CString.
withStringList :: [ByteString] -> ([CString] -> IO a) -> IO a
withStringList strings fn = go strings []
  where
    go [] cs = fn (reverse cs)
    -- TODO(fmayle): Is it worth using unsafeAsCString here?
    go (x:xs) cs = useAsCString x $ \c -> go xs (c:cs)


-- | Use a list of ByteString as an array of CString with its length.
withStringArrayLen :: [ByteString] -> ((Ptr CString, CInt) -> IO a) -> IO a
withStringArrayLen xs fn =
    withStringList xs $ \strings ->
      withArrayLen strings $ \len ptr -> fn (ptr, fromIntegral len)
