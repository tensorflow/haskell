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

{-# LANGUAGE OverloadedStrings #-}

-- | Wrapping of TensorFlow attributes into Haskell entities.
module TensorFlow.OpGen.AttrVal
       (AttrDef
       , AttrCase(..)
       , AttrTemplate(..)
       , Template
       , attrDef
       , attrOriginal
       , attrTemplate
       , templateDefault
       , templateRestrictions
       ) where

import Data.Int (Int64)
import Data.Monoid ((<>))
import Lens.Family2 (Lens', (^.))
import Lens.Family2.Unchecked (lens)
import Proto.Tensorflow.Core.Framework.AttrValue as AttrValue
import Proto.Tensorflow.Core.Framework.OpDef as OpDef
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import Proto.Tensorflow.Core.Framework.TensorShape (TensorShapeProto)
import qualified Data.ByteString as B
import qualified Data.Text as Text

-- | Specifies the optional default value and a set of allowed values
-- for the given type.
data Template a = Template {
    _templateDefault      :: Maybe a
    -- ^ The default value (mandatory if unspecified)
  , _templateRestrictions :: [a]
    -- ^ The allowed set of values, empty if no restrictions
 }

templateDefault :: Lens' (Template a) (Maybe a)
templateDefault = lens _templateDefault (\g x -> g { _templateDefault = x })

templateRestrictions :: Lens' (Template a) [a]
templateRestrictions = lens _templateRestrictions
                            (\g x -> g { _templateRestrictions = x })

data UnusedTensor

data AttrCase f
  = AttrBytes (f B.ByteString)          -- bytes s = 2; // "string"
  | AttrInt64 (f Int64)                 -- int64 i = 3; // "int"
  | AttrFloat (f Float)                 -- float f = 4; // "float"
  | AttrBool  (f Bool)                  -- bool b = 5;  // "bool"
  | AttrType  (f DataType)              -- type = 6; // "type"
    -- To be translated into TensorFlow.Types.Shape before use.
    -- Leaving as a proto to reduce dependencies.
  | AttrShape (f TensorShapeProto)      -- shape = 7; // "shape"

-- | Type-reified representation of TensorFlow AttrDef.
-- Initially limited to just the types in Op descriptors.
data AttrTemplate
  = AttrSingle (AttrCase Template)
  | AttrList (AttrCase [])
  | AttrTensor UnusedTensor         -- tensor = 8; // "tensor"

data AttrDef = AttrDef {
    _attrOriginal :: OpDef'AttrDef -- ^ the proto this value was created from
  , _attrTemplate :: AttrTemplate  -- ^ the type of the attribute
  }

attrTemplate :: Lens' AttrDef AttrTemplate
attrTemplate = lens _attrTemplate (\g x -> g { _attrTemplate = x })

attrOriginal :: Lens' AttrDef OpDef'AttrDef
attrOriginal = lens _attrOriginal (\g x -> g { _attrOriginal = x })

attrDef :: OpDef'AttrDef -> AttrDef
attrDef a = AttrDef a
                  $ translate (a^.OpDef.type')
                              (a^.OpDef.defaultValue)
                              (a^.allowedValues)

-- | Converts the given AttrValue with the type given by the string
-- into the AttrVal if the type is known.
translate :: Text.Text  -- ^ one of the TensorFlow type strings
          -> AttrValue  -- ^ default value
          -> AttrValue  -- ^ allowed values
          -> AttrTemplate
translate t defaults allowed
  | t == "string" = makeVal AttrBytes maybe's s
  | t == "int" = makeVal AttrInt64 maybe'i i
  | t == "float" = makeVal AttrFloat maybe'f f
  | t == "bool" = makeVal AttrBool maybe'b b
  | t == "type" = makeVal AttrType AttrValue.maybe'type' AttrValue.type'
  | t == "shape" = makeVal AttrShape maybe'shape shape
  | t == "tensor" = AttrTensor $ error "tensor is unimplemented"
  | t == "list(string)" = makeList AttrBytes $ list.s
  | t == "list(int)" = makeList AttrInt64 $ list.i
  | t == "list(float)" = makeList AttrFloat $ list.f
  | t == "list(bool)" = makeList AttrBool $ list.b
  | t == "list(type)" = makeList AttrType $ list.AttrValue.type'
  | t == "list(shape)" = makeList AttrShape $ list.shape
  | t == "list(tensor)" = AttrTensor $ error "list(tensor) is unimplemented"
  | t == "func" = AttrTensor $ error "func is unimplemented"
  | otherwise = error $ show ("Unknown attribute type " <> t) ++
                        "," ++ show defaults ++
                        "," ++ show allowed
  where makeVal c x y = AttrSingle $ c $
                        Template (defaults^.x) (allowed^.list.y)
        makeList c y  = AttrList $ c $ defaults^.y
