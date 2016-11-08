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

{-# LANGUAGE ExplicitNamespaces #-}

-- | The core functionality of TensorFlow.
--
-- Unless you are defining ops, you do not need to import other modules from
-- this package.
--
-- Ops are provided in the following modules:
--
--     * "TensorFlow.Ops"
--     * "TensorFlow.GenOps.Core"
--     * "TensorFlow.Gradient"
--     * "TensorFlow.EmbeddingOps"
module TensorFlow.Core
    ( -- * Session
      Session
    , SessionOption
    , sessionConfig
    , sessionTarget
    , runSession
    , runSessionWithOptions
      -- ** Building graphs
    , build
    , buildAnd
    , buildWithSummary
      -- ** Running graphs
    , Fetchable
    , Scalar(..)
    , Nodes
    , run
    , run_
    , Feed
    , feed
    , runWithFeeds
    , runWithFeeds_
      -- ** Async
    , asyncProdNodes

      -- * Build
    , Build
    , BuildT
    , render
    , asGraphDef
    , addGraphDef

      -- * Tensor
    , ControlNode
    , Tensor
    , Value
    , Ref
    , TensorKind(..)
    , tensorAttr
    , value
    , tensorFromName
      -- ** Element types
    , TensorData
    , TensorType(decodeTensorData, encodeTensorData)
    , Shape(..)
    , OneOf
    , type (/=)

      -- * Op combinators
    , colocateWith
    , Device(..)
    , withDevice
    , withNameScope
    , named
      -- ** Dependencies
    , withControlDependencies
    , group
      -- ** Misc
    , identity
    , noOp
    ) where

import TensorFlow.Build
import TensorFlow.ControlFlow
import TensorFlow.Nodes
import TensorFlow.Output
import TensorFlow.Session
import TensorFlow.Tensor
import TensorFlow.Types
