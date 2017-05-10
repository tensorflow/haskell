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
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Queues in TensorFlow graph. Very limited support for now.
module TensorFlow.Queue (Queue, makeQueue, enqueue, dequeue) where

import Data.ByteString (ByteString)
import Data.Int (Int64)
import Data.Proxy (Proxy(..))
import Lens.Family2 ((.~), (&))
import TensorFlow.Build (ControlNode, MonadBuild, build, addInitializer, opAttr, opDef)
import TensorFlow.BuildOp (buildOp)
import TensorFlow.ControlFlow (group)
import qualified TensorFlow.GenOps.Core as CoreOps
import TensorFlow.Tensor (Ref, Value, Tensor, TensorList)
import TensorFlow.Types (TensorTypes, fromTensorTypes)

-- | A queue carrying tuples.
data Queue (as :: [*]) = Queue { handle :: Handle }

type Handle = Tensor Ref ByteString

-- | Adds the given values to the queue.
enqueue :: forall as v m . (MonadBuild m, TensorTypes as)
           => Queue as
           -> TensorList v as
           -> m ControlNode
enqueue = CoreOps.queueEnqueue . handle

-- | Retrieves the values from the queue.
dequeue :: forall as m . (MonadBuild m, TensorTypes as)
           => Queue as
           -> m (TensorList Value as)
           -- ^ Dequeued tensors. They are coupled in a sense
           -- that values appear together, even if they are
           -- not consumed together.
dequeue = CoreOps.queueDequeue . handle

-- | Creates a new queue with the given capacity and shared name.
makeQueue :: forall as m . (MonadBuild m, TensorTypes as)
              => Int64  -- ^ The upper bound on the number of elements in
                        --  this queue. Negative numbers mean no limit.
              -> ByteString -- ^ If non-empty, this queue will be shared
                            -- under the given name across multiple sessions.
              -> m (Queue as)
makeQueue capacity sharedName = do
    q <- build $ buildOp [] (opDef "FIFOQueue"
                     & opAttr "component_types" .~ fromTensorTypes (Proxy :: Proxy as)
                     & opAttr "shared_name" .~ sharedName
                     & opAttr "capacity" .~ capacity
                    )
    group q >>= addInitializer
    return (Queue q)

-- TODO(gnezdo): Figure out the closing story for queues.
