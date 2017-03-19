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
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module TensorFlow.Tensor where

import Data.String (IsString(..))
import qualified Data.Text as Text
import Lens.Family2 (Lens', (^.))
import Lens.Family2.Unchecked (lens)

import TensorFlow.Output (Output)
import TensorFlow.Types
    ( TensorData(..)
    , ListOf(..)
    )
import qualified TensorFlow.Internal.FFI as FFI

-- | A named output of a TensorFlow operation.
--
-- The type parameter @a@ is the type of the elements in the 'Tensor'.  The
-- parameter @v@ is either 'Value' or 'Ref', depending on whether the graph is
-- treating this op output as an immutable 'Value' or a stateful 'Ref' (e.g., a
-- variable).  Note that a @Tensor Ref@ can be casted into a @Tensor Value@ via
-- 'value'.
data Tensor v a = Tensor (TensorKind v) Output

data Value
data Ref

-- | This class provides a runtime switch on whether a 'Tensor' should be
-- treated as a 'Value' or as a 'Ref'.
data TensorKind v where
  ValueKind :: TensorKind Value
  RefKind :: TensorKind Ref

tensorKind :: Lens' (Tensor v a) (TensorKind v)
tensorKind = lens (\(Tensor v _) -> v) (\(Tensor _ o) v -> Tensor v o)

tensorOutput :: Lens' (Tensor v a) Output
tensorOutput = lens (\(Tensor _ o) -> o) (\(Tensor v _) o -> Tensor v o)

-- | Cast a 'Tensor *' into a 'Tensor Value'. Common usage is to cast a
-- Ref into Value. This behaves like a no-op.
value :: Tensor v a -> Tensor Value a
value (Tensor _ o) = Tensor ValueKind o

-- | A pair of a 'Tensor' and some data that should be fed into that 'Tensor'
-- when running the graph.
data Feed = Feed Output FFI.TensorData

-- | Create a 'Feed' for feeding the given data into a 'Tensor' when running
-- the graph.
--
-- Note that if a 'Tensor' is rendered, its identity may change; so feeding the
-- rendered 'Tensor' may be different than feeding the original 'Tensor'.
feed :: Tensor v a -> TensorData a -> Feed
feed (Tensor _ o) (TensorData td) = Feed o td

-- | Create a 'Tensor' for a given name.  This can be used to reference nodes
-- in a 'GraphDef' that was loaded via 'addGraphDef'.
-- TODO(judahjacobson): add more safety checks here.
tensorFromName :: TensorKind v -> Text.Text -> Tensor v a
tensorFromName v = Tensor v . fromString . Text.unpack

type TensorList v = ListOf (Tensor v)

tensorListOutputs :: TensorList v as -> [Output]
tensorListOutputs Nil = []
tensorListOutputs (t :/ ts) = (t ^. tensorOutput) : tensorListOutputs ts
