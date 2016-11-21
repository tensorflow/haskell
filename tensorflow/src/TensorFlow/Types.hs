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

{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
-- We use UndecidableInstances for type families with recursive definitions
-- like "\\".  Those instances will terminate since each equation unwraps one
-- cons cell of a type-level list.
{-# LANGUAGE UndecidableInstances #-}

module TensorFlow.Types
    ( TensorType(..)
    , TensorData(..)
    , Shape(..)
    , protoShape
    , Attribute(..)
    -- * Type constraints
    , OneOf
    , type (/=)
    -- ** Implementation of constraints
    , TypeError
    , ExcludedCase
    , TensorTypes
    , NoneOf
    , type (\\)
    , Delete
    , AllTensorTypes
    ) where

import Data.Complex (Complex)
import Data.Default (def)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Monoid ((<>))
import Data.Word (Word8, Word16, Word64)
import Foreign.Storable (Storable)
import GHC.Exts (Constraint, IsList(..))
import Lens.Family2 (Lens', view, (&), (.~))
import Lens.Family2.Unchecked (iso)
import qualified Data.Attoparsec.ByteString as Atto
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import Data.ByteString.Builder (Builder)
import qualified Data.ByteString.Builder as Builder
import qualified Data.ByteString.Lazy as L
import qualified Data.Vector as V
import qualified Data.Vector.Storable as S
import Proto.Tensorflow.Core.Framework.AttrValue
    ( AttrValue(..)
    , AttrValue'ListValue(..)
    , b
    , f
    , i
    , s
    , list
    , type'
    , shape
    , tensor
    )
import Proto.Tensorflow.Core.Framework.Tensor as Tensor
    ( TensorProto(..)
    , floatVal
    , doubleVal
    , intVal
    , stringVal
    , int64Val
    , stringVal
    , boolVal
    )
import Proto.Tensorflow.Core.Framework.TensorShape
    ( TensorShapeProto(..)
    , dim
    , size
    )
import Proto.Tensorflow.Core.Framework.Types (DataType(..))

import TensorFlow.Internal.VarInt (getVarInt, putVarInt)
import qualified TensorFlow.Internal.FFI as FFI

-- | Data about a tensor that is encoded for the TensorFlow APIs.
newtype TensorData a = TensorData { unTensorData :: FFI.TensorData }

-- | The class of scalar types supported by tensorflow.
class TensorType a where
    tensorType :: a -> DataType
    tensorRefType :: a -> DataType
    tensorVal :: Lens' TensorProto [a]
    -- | Decode the bytes of a TensorData into a Vector.
    decodeTensorData :: TensorData a -> V.Vector a
    -- | Encode a Vector into a TensorData.
    --
    -- The values should be in row major order, e.g.,
    --
    --   element 0:   index (0, ..., 0)
    --   element 1:   index (0, ..., 1)
    --   ...
    encodeTensorData :: Shape -> V.Vector a -> TensorData a

-- All types, besides ByteString, are encoded as simple arrays and we can use
-- Vector.Storable to encode/decode by type casting pointers.

-- TODO(fmayle): Assert that the data type matches the return type.
simpleDecode :: Storable a => TensorData a -> V.Vector a
simpleDecode = S.convert . S.unsafeCast . FFI.tensorDataBytes . unTensorData

simpleEncode :: forall a . (TensorType a, Storable a)
             => Shape -> V.Vector a -> TensorData a
simpleEncode (Shape xs)
    = TensorData . FFI.TensorData xs dt . S.unsafeCast . S.convert
  where
    dt = tensorType (undefined :: a)

instance TensorType Float where
    tensorType _ = DT_FLOAT
    tensorRefType _ = DT_FLOAT_REF
    tensorVal = floatVal
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Double where
    tensorType _ = DT_DOUBLE
    tensorRefType _ = DT_DOUBLE_REF
    tensorVal = doubleVal
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Int32 where
    tensorType _ = DT_INT32
    tensorRefType _ = DT_INT32_REF
    tensorVal = intVal
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Int64 where
    tensorType _ = DT_INT64
    tensorRefType _ = DT_INT64_REF
    tensorVal = int64Val
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

integral :: Integral a => Lens' [Int32] [a]
integral = iso (fmap fromIntegral) (fmap fromIntegral)

instance TensorType Word8 where
    tensorType _ = DT_UINT8
    tensorRefType _ = DT_UINT8_REF
    tensorVal = intVal . integral
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Word16 where
    tensorType _ = DT_UINT16
    tensorRefType _ = DT_UINT16_REF
    tensorVal = intVal . integral
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Int16 where
    tensorType _ = DT_INT16
    tensorRefType _ = DT_INT16_REF
    tensorVal = intVal . integral
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType Int8 where
    tensorType _ = DT_INT8
    tensorRefType _ = DT_INT8_REF
    tensorVal = intVal . integral
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType ByteString where
    tensorType _ = DT_STRING
    tensorRefType _ = DT_STRING_REF
    tensorVal = stringVal
    -- Encoded data layout (described in third_party/tensorflow/c/c_api.h):
    --   table offsets for each element :: [Word64]
    --   at each element offset:
    --     string length :: VarInt64
    --     string data   :: [Word8]
    -- TODO(fmayle): Benchmark these functions.
    decodeTensorData tensorData =
        either (\err -> error $ "Malformed TF_STRING tensor; " ++ err) id $
            if expected /= count
                then Left $ "decodeTensorData for ByteString count mismatch " ++
                            show (expected, count)
                else V.mapM decodeString (S.convert offsets)
      where
        expected = S.length offsets
        count = fromIntegral $ product $ FFI.tensorDataDimensions
                    $ unTensorData tensorData
        bytes = FFI.tensorDataBytes $ unTensorData tensorData
        offsets = S.take count $ S.unsafeCast bytes :: S.Vector Word64
        dataBytes = B.pack $ S.toList $ S.drop (count * 8) bytes
        decodeString :: Word64 -> Either String ByteString
        decodeString offset =
            let stringDataStart = B.drop (fromIntegral offset) dataBytes
            in Atto.eitherResult $ Atto.parse stringParser stringDataStart
        stringParser :: Atto.Parser ByteString
        stringParser = getVarInt >>= Atto.take . fromIntegral
    encodeTensorData (Shape xs) vec =
        TensorData $ FFI.TensorData xs dt byteVector
      where
        dt = tensorType (undefined :: ByteString)
        -- Add a string to an offset table and data blob.
        addString :: (Builder, Builder, Word64)
                  -> ByteString
                  -> (Builder, Builder, Word64)
        addString (table, strings, offset) str =
            ( table <> Builder.word64LE offset
            , strings <> lengthBytes <> Builder.byteString str
            , offset + lengthBytesLen + strLen
            )
          where
            strLen = fromIntegral $ B.length str
            lengthBytes = putVarInt $ fromIntegral $ B.length str
            lengthBytesLen =
                fromIntegral $ L.length $ Builder.toLazyByteString lengthBytes
        -- Encode all strings.
        (table', strings', _) = V.foldl' addString (mempty, mempty, 0) vec
        -- Concat offset table with data.
        bytes = table' <> strings'
        -- Convert to Vector Word8.
        byteVector = S.fromList $ L.unpack $ Builder.toLazyByteString bytes


instance TensorType Bool where
    tensorType _ = DT_BOOL
    tensorRefType _ = DT_BOOL_REF
    tensorVal = boolVal
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorType (Complex Float) where
    tensorType _ = DT_COMPLEX64
    tensorRefType _ = DT_COMPLEX64
    tensorVal = error "TODO (Complex Float)"
    decodeTensorData = error "TODO (Complex Float)"
    encodeTensorData = error "TODO (Complex Float)"

instance TensorType (Complex Double) where
    tensorType _ = DT_COMPLEX128
    tensorRefType _ = DT_COMPLEX128
    tensorVal = error "TODO (Complex Double)"
    decodeTensorData = error "TODO (Complex Double)"
    encodeTensorData = error "TODO (Complex Double)"

-- | Shape (dimensions) of a tensor.
newtype Shape = Shape [Int64] deriving Show

instance IsList Shape where
    type Item Shape = Int64
    fromList = Shape . fromList
    toList (Shape ss) = toList ss

protoShape :: Lens' TensorShapeProto Shape
protoShape = iso protoToShape shapeToProto
  where
    protoToShape = Shape . fmap (view size) . view dim
    shapeToProto (Shape ds) = def & dim .~ fmap (\d -> def & size .~ d) ds


class Attribute a where
    attrLens :: Lens' AttrValue a

instance Attribute Float where
    attrLens = f

instance Attribute ByteString where
    attrLens = s

instance Attribute Int64 where
    attrLens = i

instance Attribute DataType where
    attrLens = type'

instance Attribute TensorProto where
    attrLens = tensor

instance Attribute Bool where
    attrLens = b

instance Attribute Shape where
    attrLens = shape . protoShape

-- TODO(gnezdo): support generating list(Foo) from [Foo].
instance Attribute AttrValue'ListValue where
    attrLens = list

instance Attribute [DataType] where
    attrLens = list . type'

instance Attribute [Int64] where
    attrLens = list . i

-- | A 'Constraint' specifying the possible choices of a 'TensorType'.
--
-- We implement a 'Constraint' like @OneOf '[Double, Float] a@ by turning the
-- natural representation as a conjunction, i.e.,
--
-- @
--    a == Double || a == Float
-- @
--
-- into a disjunction like
--
-- @
--     a \/= Int32 && a \/= Int64 && a \/= ByteString && ...
-- @
--
-- using an enumeration of all the possible 'TensorType's.
type OneOf ts a
    = (TensorType a, TensorTypes ts, NoneOf (AllTensorTypes \\ ts) a)

-- | A 'Constraint' checking that the input is a list of 'TensorType's.
-- Helps improve error messages when using 'OneOf'.
type family TensorTypes ts :: Constraint where
    TensorTypes '[] = ()
    TensorTypes (t ': ts) = (TensorType t, TensorTypes ts)

-- | A constraint checking that two types are different.
type family a /= b :: Constraint where
    a /= a = TypeError a ~ ExcludedCase
    a /= b = ()

-- | Helper types to produce a reasonable type error message when the Constraint
-- "a /= a" fails.
-- TODO(judahjacobson): Use ghc-8's CustomTypeErrors for this.
data TypeError a
data ExcludedCase

-- | An enumeration of all valid 'TensorType's.
type AllTensorTypes =
    -- NOTE: This list should be kept in sync with
    -- TensorFlow.OpGen.dtTypeToHaskell.
    -- TODO: Add support for Complex Float/Double.
    '[ Float
     , Double
     , Int8
     , Int16
     , Int32
     , Int64
     , Word8
     , Word16
     , ByteString
     , Bool
     ]

-- | Removes a type from the given list of types.
type family Delete a as where
    Delete a '[] = '[]
    Delete a (a ': as) = Delete a as
    Delete a (b ': as) = b ': Delete a as

-- | Takes the difference of two lists of types.
type family as \\ bs where
    as \\ '[] = as
    as \\ (b ': bs) = Delete b as \\ bs

-- | A constraint that the type @a@ doesn't appear in the type list @ts@.
-- Assumes that @a@ and each of the elements of @ts@ are 'TensorType's.
type family NoneOf ts a :: Constraint where
    NoneOf '[] a = ()
    NoneOf (t ': ts) a = (a /= t, NoneOf ts a)
