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

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}  -- For the Render class

module TensorFlow.Tensor where

import Data.ByteString (ByteString)
import Data.String (IsString(..))
import qualified Data.Text as Text
import Lens.Family2 ((^.))
import Lens.Family2.State ((%=), use)

import Proto.Tensorflow.Core.Framework.NodeDef (device)
import TensorFlow.Build
import TensorFlow.Output (Output, NodeName, outputNodeName, Device(..))
import TensorFlow.Types
    ( TensorType
    , TensorData(..)
    , ListOf(..)
    )
import qualified TensorFlow.Internal.FFI as FFI

-- | A named output of a TensorFlow operation.
--
-- The type parameter @a@ is the type of the elements in the 'Tensor'.  The
-- parameter @v@ is either:
--
--   * 'Build': An unrendered, immutable value.
--   * 'Value': A rendered, immutable value.
--   * 'Ref': A rendered stateful handle (e.g., a variable).
--
-- Note that 'expr', 'value', 'render' and 'renderValue' can help convert between
-- the different types of 'Tensor'.
data Tensor v a where
    Tensor :: TensorKind v => {tensorOutput :: v Output} -> Tensor v a

newtype Value a = Value {runValue :: a}
    deriving Functor

instance Applicative Value where
    pure = Value
    Value f <*> Value x = Value $ f x

instance Monad Value where
    f >>= g = g $ runValue f

newtype Ref a = Ref {runRef :: a}
    deriving Functor

instance Applicative Ref where
    pure = Ref
    Ref f <*> Ref x = Ref $ f x

instance Monad Ref where
    f >>= g = g $ runRef f

-- | Cast a 'Tensor Ref' into a 'Tensor Value'. This behaves like a no-op.
value :: Tensor Ref a -> Tensor Value a
value (Tensor o) = Tensor $ Value $ runRef o

renderValue :: MonadBuild m => Tensor v a -> m (Tensor Value a)
renderValue (Tensor o) = render $ Tensor $ toBuild o

-- | A pair of a 'Tensor' and some data that should be fed into that 'Tensor'
-- when running the graph.
data Feed = Feed Output FFI.TensorData

-- | A class ensuring that a given tensor is rendered, i.e., has a fixed
-- name, device, etc.
class Rendered t where
    renderedOutput :: t a -> Output

instance Rendered (Tensor Value) where
    renderedOutput = runValue . tensorOutput

instance Rendered (Tensor Ref) where
    renderedOutput = runRef . tensorOutput

tensorNodeName :: Rendered t => t a -> NodeName
tensorNodeName = outputNodeName . renderedOutput


-- | Create a 'Feed' for feeding the given data into a 'Tensor' when running
-- the graph.
--
-- Note that if a 'Tensor' is rendered, its identity may change; so feeding the
-- rendered 'Tensor' may be different than feeding the original 'Tensor'.
feed :: Rendered t => t a -> TensorData a -> Feed
feed t (TensorData td) = Feed (renderedOutput t) td

-- | Create a 'Tensor' for a given name.  This can be used to reference nodes
-- in a 'GraphDef' that was loaded via 'addGraphDef'.
-- TODO(judahjacobson): add more safety checks here.
tensorFromName :: TensorKind v => Text.Text -> Tensor v a
tensorFromName = Tensor . pure . fromString . Text.unpack

-- | Like 'tensorFromName', but type-restricted to 'Value'.
tensorValueFromName :: Text.Text -> Tensor Value a
tensorValueFromName = tensorFromName

-- | Like 'tensorFromName', but type-restricted to 'Ref'.
tensorRefFromName :: Text.Text -> Tensor Ref a
tensorRefFromName = tensorFromName

type TensorList v = ListOf (Tensor v)

tensorListOutputs :: Rendered (Tensor v) => TensorList v as -> [Output]
tensorListOutputs Nil = []
tensorListOutputs (t :/ ts) = renderedOutput t : tensorListOutputs ts

-- | Places all nodes rendered in the given 'Build' action on the same
-- device as the given Tensor (see also 'withDevice'). Make sure that
-- the action has side effects of rendering the desired tensors. A pure
-- return would not have the desired effect.
colocateWith :: (MonadBuild m, Rendered t) => t b -> m a -> m a
colocateWith t x = do
    d <- build $ Device . (^. device)
               <$> lookupNode (outputNodeName $ renderedOutput t)
    withDevice (Just d) x


-- | Render a 'Tensor', fixing its name, scope, device and control inputs from
-- the 'MonadBuild' context.  Also renders any dependencies of the 'Tensor' that
-- weren't already rendered.
--
-- This operation is idempotent; calling 'render' on the same input in the same
-- context will produce the same result.  However, rendering the same
-- @Tensor Build@ in two different contexts may result in two different
-- @Tensor Value@s.
render :: MonadBuild m => Tensor Build a -> m (Tensor Value a)
render (Tensor t) = Tensor . Value <$> build t

-- TODO: better name.
expr :: TensorKind v => Tensor v a -> Tensor Build a
expr (Tensor o) = Tensor $ toBuild o

-- | Records the given summary action in Build for retrieval with
-- Summary protocol buffer in string form. For safety, use the
-- pre-composed functions: Logging.scalarSummary and
-- Logging.histogramSummary.
addSummary :: (MonadBuild m, TensorKind v) => Tensor v ByteString -- ^ A 'SummaryTensor'
                        -> m ()
addSummary t = build $ do
    -- TODO: more generic way
    o <- toBuild $ tensorOutput t
    summaries %= (o :)

-- | Retrieves the summary ops collected thus far. Typically this only
-- happens once, but if 'TensorFlow.Session.buildWithSummary' is used
-- repeatedly, the values accumulate.
collectAllSummaries :: MonadBuild m => m [SummaryTensor]
collectAllSummaries = build $ map (Tensor . Value) <$> use summaries

-- | Synonym for the tensors that return serialized Summary proto.
type SummaryTensor = Tensor Value ByteString

-- | An internal class for kinds of Tensors.
class Monad v => TensorKind v where
    toBuild :: v a -> Build a

instance TensorKind Value where
    toBuild = return . runValue

instance TensorKind Ref where
    toBuild = return . runRef

instance TensorKind Build where
    toBuild = id


-- | Types which can be converted to `Tensor`.
class ToTensor t where
    toTensor :: TensorType a => t a -> Tensor Build a

instance TensorKind v => ToTensor (Tensor v) where
    toTensor = expr
