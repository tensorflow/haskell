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


{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- Orphan instances for certain proto messages/enums, used internally.
-- TODO(judahjacobson): consider making proto-lens generate some or all of
-- these automatically; or, alternately, make new Haskell datatypes.
module TensorFlow.Orphans() where

import Proto.Tensorflow.Core.Framework.AttrValue
    ( AttrValue(..)
    , AttrValue'ListValue(..)
    , NameAttrList(..)
    )
import Proto.Tensorflow.Core.Framework.NodeDef
    ( NodeDef(..))
import Proto.Tensorflow.Core.Framework.ResourceHandle
    ( ResourceHandle(..))
import Proto.Tensorflow.Core.Framework.Tensor
    (TensorProto(..))
import Proto.Tensorflow.Core.Framework.TensorShape
    (TensorShapeProto(..), TensorShapeProto'Dim(..))
import Proto.Tensorflow.Core.Framework.Types (DataType(..))

deriving instance Ord AttrValue
deriving instance Ord AttrValue'ListValue
deriving instance Ord DataType
deriving instance Ord NameAttrList
deriving instance Ord NodeDef
deriving instance Ord ResourceHandle
deriving instance Ord TensorProto
deriving instance Ord TensorShapeProto
deriving instance Ord TensorShapeProto'Dim
