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
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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
    , TensorDataType(..)
    , Scalar(..)
    , Shape(..)
    , protoShape
    , Attribute(..)
    , DataType(..)
    , ResourceHandle
    -- * Lists
    , ListOf(..)
    , List
    , (/:/)
    , TensorTypeProxy(..)
    , TensorTypes(..)
    , TensorTypeList
    , fromTensorTypeList
    , fromTensorTypes
    -- * Type constraints
    , OneOf
    , type (/=)
    , OneOfs
    -- ** Implementation of constraints
    , TypeError
    , ExcludedCase
    , NoneOf
    , type (\\)
    , Delete
    , AllTensorTypes
    ) where

import Data.Functor.Identity (Identity(..))
import Data.Complex (Complex)
import Data.Default (def)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Monoid ((<>))
import Data.Proxy (Proxy(..))
import Data.String (IsString)
import Data.Word (Word8, Word16, Word64)
import Foreign.Storable (Storable)
import GHC.Exts (Constraint, IsList(..))
import Lens.Family2 (Lens', view, (&), (.~))
import Lens.Family2.Unchecked (iso)
import Text.Printf (printf)
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
import Proto.Tensorflow.Core.Framework.ResourceHandle
    (ResourceHandle)
import Proto.Tensorflow.Core.Framework.Tensor as Tensor
    ( TensorProto(..)
    , boolVal
    , doubleVal
    , floatVal
    , intVal
    , int64Val
    , resourceHandleVal
    , stringVal
    , stringVal
    )
import Proto.Tensorflow.Core.Framework.TensorShape
    ( TensorShapeProto(..)
    , dim
    , size
    )
import Proto.Tensorflow.Core.Framework.Types (DataType(..))

import TensorFlow.Internal.VarInt (getVarInt, putVarInt)
import qualified TensorFlow.Internal.FFI as FFI

-- | The class of scalar types supported by tensorflow.
class TensorType a where
    tensorType :: a -> DataType
    tensorRefType :: a -> DataType
    tensorVal :: Lens' TensorProto [a]

instance TensorType Float where
    tensorType _ = DT_FLOAT
    tensorRefType _ = DT_FLOAT_REF
    tensorVal = floatVal

instance TensorType Double where
    tensorType _ = DT_DOUBLE
    tensorRefType _ = DT_DOUBLE_REF
    tensorVal = doubleVal

instance TensorType Int32 where
    tensorType _ = DT_INT32
    tensorRefType _ = DT_INT32_REF
    tensorVal = intVal

instance TensorType Int64 where
    tensorType _ = DT_INT64
    tensorRefType _ = DT_INT64_REF
    tensorVal = int64Val

integral :: Integral a => Lens' [Int32] [a]
integral = iso (fmap fromIntegral) (fmap fromIntegral)

instance TensorType Word8 where
    tensorType _ = DT_UINT8
    tensorRefType _ = DT_UINT8_REF
    tensorVal = intVal . integral

instance TensorType Word16 where
    tensorType _ = DT_UINT16
    tensorRefType _ = DT_UINT16_REF
    tensorVal = intVal . integral

instance TensorType Int16 where
    tensorType _ = DT_INT16
    tensorRefType _ = DT_INT16_REF
    tensorVal = intVal . integral

instance TensorType Int8 where
    tensorType _ = DT_INT8
    tensorRefType _ = DT_INT8_REF
    tensorVal = intVal . integral

instance TensorType ByteString where
    tensorType _ = DT_STRING
    tensorRefType _ = DT_STRING_REF
    tensorVal = stringVal

instance TensorType Bool where
    tensorType _ = DT_BOOL
    tensorRefType _ = DT_BOOL_REF
    tensorVal = boolVal

instance TensorType (Complex Float) where
    tensorType _ = DT_COMPLEX64
    tensorRefType _ = DT_COMPLEX64
    tensorVal = error "TODO (Complex Float)"

instance TensorType (Complex Double) where
    tensorType _ = DT_COMPLEX128
    tensorRefType _ = DT_COMPLEX128
    tensorVal = error "TODO (Complex Double)"

instance TensorType ResourceHandle where
    tensorType _ = DT_RESOURCE
    tensorRefType _ = DT_RESOURCE_REF
    tensorVal = resourceHandleVal

-- | Tensor data with the correct memory layout for tensorflow.
newtype TensorData a = TensorData { unTensorData :: FFI.TensorData }

-- | Types that can be converted to and from 'TensorData'.
--
-- 'S.Vector' is the most efficient to encode/decode for most element types.
class TensorType a => TensorDataType s a where
    -- | Decode the bytes of a 'TensorData' into an 's'.
    decodeTensorData :: TensorData a -> s a
    -- | Encode an 's' into a 'TensorData'.
    --
    -- The values should be in row major order, e.g.,
    --
    --   element 0:   index (0, ..., 0)
    --   element 1:   index (0, ..., 1)
    --   ...
    encodeTensorData :: Shape -> s a -> TensorData a

-- All types, besides ByteString and Bool, are encoded as simple arrays and we
-- can use Vector.Storable to encode/decode by type casting pointers.

-- TODO(fmayle): Assert that the data type matches the return type.
simpleDecode :: Storable a => TensorData a -> S.Vector a
simpleDecode = S.unsafeCast . FFI.tensorDataBytes . unTensorData

simpleEncode :: forall a . (TensorType a, Storable a)
             => Shape -> S.Vector a -> TensorData a
simpleEncode (Shape xs) v =
    if product xs /= fromIntegral (S.length v)
        then error $ printf
            "simpleEncode: bad vector length for shape %v: expected=%d got=%d"
            (show xs) (product xs) (S.length v)
        else TensorData (FFI.TensorData xs dt (S.unsafeCast v))
  where
    dt = tensorType (undefined :: a)

instance TensorDataType S.Vector Float where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Double where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Int8 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Int16 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Int32 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Int64 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Word8 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

instance TensorDataType S.Vector Word16 where
    decodeTensorData = simpleDecode
    encodeTensorData = simpleEncode

-- TODO: Haskell and tensorflow use different byte sizes for bools, which makes
-- encoding more expensive. It may make sense to define a custom boolean type.
instance TensorDataType S.Vector Bool where
    decodeTensorData =
        S.convert . S.map (/= 0) . FFI.tensorDataBytes . unTensorData
    encodeTensorData (Shape xs) =
        TensorData . FFI.TensorData xs DT_BOOL . S.map fromBool . S.convert
      where
        fromBool x = if x then 1 else 0 :: Word8

instance {-# OVERLAPPABLE #-} (Storable a, TensorDataType S.Vector a, TensorType a)
    => TensorDataType V.Vector a where
    decodeTensorData = (S.convert :: S.Vector a -> V.Vector a) . decodeTensorData
    encodeTensorData x = encodeTensorData x . (S.convert :: V.Vector a -> S.Vector a)

instance {-# OVERLAPPING #-} TensorDataType V.Vector (Complex Float) where
    decodeTensorData = error "TODO (Complex Float)"
    encodeTensorData = error "TODO (Complex Float)"

instance {-# OVERLAPPING #-} TensorDataType V.Vector (Complex Double) where
    decodeTensorData = error "TODO (Complex Double)"
    encodeTensorData = error "TODO (Complex Double)"

instance {-# OVERLAPPING #-} TensorDataType V.Vector ByteString where
    -- Encoded data layout (described in third_party/tensorflow/c/c_api.h):
    --   table offsets for each element :: [Word64]
    --   at each element offset:
    --     string length :: VarInt64
    --     string data   :: [Word8]
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

newtype Scalar a = Scalar {unScalar :: a}
    deriving (Show, Eq, Ord, Num, Fractional, Floating, Real, RealFloat,
              RealFrac, IsString)

instance (TensorDataType V.Vector a, TensorType a) => TensorDataType Scalar a where
    decodeTensorData = Scalar . headFromSingleton . decodeTensorData
    encodeTensorData x (Scalar y) = encodeTensorData x (V.fromList [y])

headFromSingleton :: V.Vector a -> a
headFromSingleton x
    | V.length x == 1 = V.head x
    | otherwise = error $
                  "Unable to extract singleton from tensor of length "
                  ++ show (V.length x)


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
    shapeToProto (Shape ds) = (def :: TensorShapeProto) & dim .~ fmap (\d -> def & size .~ d) ds


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

-- | A heterogeneous list type.
data ListOf f as where
    Nil :: ListOf f '[]
    (:/) :: f a -> ListOf f as -> ListOf f (a ': as)

infixr 5 :/

type family All f as :: Constraint where
    All f '[] = ()
    All f (a ': as) = (f a, All f as)

type family Map f as where
    Map f '[] = '[]
    Map f (a ': as) = f a ': Map f as

instance All Eq (Map f as) => Eq (ListOf f as) where
    Nil == Nil = True
    (x :/ xs) == (y :/ ys) = x == y && xs == ys
    -- Newer versions of GHC use the GADT to tell that the previous cases are
    -- exhaustive.
#if __GLASGOW_HASKELL__ < 800
    _ == _ = False
#endif

instance All Show (Map f as) => Show (ListOf f as) where
    showsPrec _ Nil = showString "Nil"
    showsPrec d (x :/ xs) = showParen (d > 10)
                                $ showsPrec 6 x . showString " :/ "
                                    . showsPrec 6 xs

type List = ListOf Identity

-- | Equivalent of ':/' for lists.
(/:/) :: a -> List as -> List (a ': as)
(/:/) = (:/) . Identity

infixr 5 /:/

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
    -- Assert `TensorTypes' ts` to make error messages a little better.
    = (TensorType a, TensorTypes' ts, NoneOf (AllTensorTypes \\ ts) a)

type OneOfs ts as = (TensorTypes as, TensorTypes' ts,
                        NoneOfs (AllTensorTypes \\ ts) as)

type family NoneOfs ts as :: Constraint where
    NoneOfs ts '[] = ()
    NoneOfs ts (a ': as) = (NoneOf ts a, NoneOfs ts as)

data TensorTypeProxy a where
    TensorTypeProxy :: TensorType a => TensorTypeProxy a

type TensorTypeList = ListOf TensorTypeProxy

fromTensorTypeList :: TensorTypeList ts -> [DataType]
fromTensorTypeList Nil = []
fromTensorTypeList ((TensorTypeProxy :: TensorTypeProxy t) :/ ts)
    = tensorType (undefined :: t) : fromTensorTypeList ts

fromTensorTypes :: forall as . TensorTypes as => Proxy as -> [DataType]
fromTensorTypes _ = fromTensorTypeList (tensorTypes :: TensorTypeList as)

class TensorTypes (ts :: [*]) where
    tensorTypes :: TensorTypeList ts

instance TensorTypes '[] where
    tensorTypes = Nil

-- | A constraint that the input is a list of 'TensorTypes'.
instance (TensorType t, TensorTypes ts) => TensorTypes (t ': ts) where
    tensorTypes = TensorTypeProxy :/ tensorTypes

-- | A simpler version of the 'TensorTypes' class, that doesn't run
-- afoul of @-Wsimplifiable-class-constraints@.
--
-- In more detail: the constraint @OneOf '[Double, Float] a@ leads
-- to the constraint @TensorTypes' '[Double, Float]@, as a safety-check
-- to give better error messages.  However, if @TensorTypes'@ were a class,
-- then GHC 8.2.1 would complain with the above warning unless @NoMonoBinds@
-- were enabled.  So instead, we use a separate type family for this purpose.
-- For more details: https://ghc.haskell.org/trac/ghc/ticket/11948
type family TensorTypes' (ts :: [*]) :: Constraint where
    -- Specialize this type family when `ts` is a long list, to avoid deeply
    -- nested tuples of constraints.  Works around a bug in ghc-8.0:
    -- https://ghc.haskell.org/trac/ghc/ticket/12175
    TensorTypes' (t1 ': t2 ': t3 ': t4 ': ts)
        = (TensorType t1, TensorType t2, TensorType t3, TensorType t4
              , TensorTypes' ts)
    TensorTypes' (t1 ': t2 ': t3 ': ts)
        = (TensorType t1, TensorType t2, TensorType t3, TensorTypes' ts)
    TensorTypes' (t1 ': t2 ': ts)
        = (TensorType t1, TensorType t2, TensorTypes' ts)
    TensorTypes' (t ': ts) = (TensorType t, TensorTypes' ts)
    TensorTypes' '[] = ()

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
    -- Specialize this type family when `ts` is a long list, to avoid deeply
    -- nested tuples of constraints.  Works around a bug in ghc-8.0:
    -- https://ghc.haskell.org/trac/ghc/ticket/12175
    NoneOf (t1 ': t2 ': t3 ': t4 ': ts) a
        = (a /= t1, a /= t2, a /= t3, a /= t4, NoneOf ts a)
    NoneOf (t1 ': t2 ': t3 ': ts) a = (a /= t1, a /= t2, a /= t3, NoneOf ts a)
    NoneOf (t1 ': t2 ': ts) a = (a /= t1, a /= t2, NoneOf ts a)
    NoneOf (t1 ': ts) a = (a /= t1, NoneOf ts a)
    NoneOf '[] a = ()
