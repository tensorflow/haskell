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

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Output
    ( ControlNode(..)
    , Device(..)
    -- * Ops
    , NodeName(..)
    , OpDef(..)
    , opName
    , opType
    , opAttr
    , opInputs
    , opControlInputs
    , OpType(..)
    , OutputIx(..)
    , Output(..)
    , output
    , PendingNodeName(..)
    )  where

import qualified Data.Map.Strict as Map
import Data.String (IsString(..))
import Data.Text (Text)
import qualified Data.Text as Text
import Lens.Family2 (Lens')
import Lens.Family2.Unchecked (lens)
import Proto.Tensorflow.Core.Framework.AttrValue (AttrValue(..))
import Data.Default (def)
import TensorFlow.Types (Attribute, attrLens)

-- | A type of graph node which has no outputs. These nodes are
-- valuable for causing side effects when they are run.
newtype ControlNode = ControlNode { unControlNode :: NodeName }

-- | The type of op of a node in the graph.  This corresponds to the proto field
-- NodeDef.op.
newtype OpType = OpType { unOpType :: Text }
    deriving (Eq, Ord, Show)

instance IsString OpType where
    fromString = OpType . Text.pack

-- | An output of a TensorFlow node.
data Output = Output {outputIndex :: !OutputIx, outputNodeName :: !NodeName}
    deriving (Eq, Ord, Show)

output :: OutputIx -> NodeName -> Output
output = Output

newtype OutputIx = OutputIx { unOutputIx :: Int }
    deriving (Eq, Ord, Num, Enum, Show)

-- | A device that a node can be assigned to.
-- There's a naming convention where the device names
-- are constructed from job and replica names.
newtype Device = Device {deviceName :: Text}
    deriving (Eq, Ord, IsString)

instance Show Device where
    show (Device d) = show d

-- | Op definition. This corresponds somewhat to the 'NodeDef' proto.
data OpDef = OpDef
    { _opName :: !PendingNodeName
    , _opType :: !OpType
    , _opAttrs :: !(Map.Map Text AttrValue)
    , _opInputs :: [Output]
    , _opControlInputs :: [NodeName]
    }  deriving (Eq, Ord)

-- | The name specified for an unrendered Op.  If an Op has an
-- ImplicitName, it will be assigned based on the opType plus a
-- unique identifier.  Does not contain the "scope" prefix.
data PendingNodeName = ExplicitName !Text | ImplicitName
    deriving (Eq, Ord, Show)

instance IsString PendingNodeName where
    fromString = ExplicitName . fromString

-- | The name of a node in the graph.  This corresponds to the proto field
-- NodeDef.name.  Includes the scope prefix (if any) and a unique identifier
-- (if the node was implicitly named).
newtype NodeName = NodeName { unNodeName :: Text }
    deriving (Eq, Ord, Show)

opName :: Lens' OpDef PendingNodeName
opName = lens _opName (\o x -> o {_opName = x})

opType :: Lens' OpDef OpType
opType = lens _opType (\o x -> o { _opType = x})

opAttr :: Attribute a => Text -> Lens' OpDef a
opAttr n = lens _opAttrs (\o x -> o {_opAttrs = x})
              . lens (Map.findWithDefault def n) (flip (Map.insert n))
              . attrLens

opInputs :: Lens' OpDef [Output]
opInputs = lens _opInputs (\o x -> o {_opInputs = x})

opControlInputs :: Lens' OpDef [NodeName]
opControlInputs = lens _opControlInputs (\o x -> o {_opControlInputs = x})

-- TODO(gnezdo): IsString instance is weird and we should move that
-- code into a Build function
instance IsString Output where
    fromString s = case break (==':') s of
        (n, ':':ixStr) | [(ix, "" :: String)] <- read ixStr
                         -> Output (fromInteger ix) $ assigned n
        _ -> Output 0 $ assigned s
     where assigned = NodeName . Text.pack
