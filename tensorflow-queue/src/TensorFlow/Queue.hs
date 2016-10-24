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
{-# LANGUAGE ScopedTypeVariables #-}

-- | Queues in TensorFlow graph. Very limited support for now.
module TensorFlow.Queue (Queue2, makeQueue2, enqueue, dequeue) where

import Data.ByteString (ByteString)
import Data.Int (Int64)
import Lens.Family2 ((.~), (&))
import TensorFlow.Build (ControlNode, Build, addInitializer, opAttr, opDef)
import TensorFlow.BuildOp (buildOp)
import TensorFlow.ControlFlow (group)
import TensorFlow.Tensor (Ref, Tensor)
import TensorFlow.Types (TensorType, tensorType)

-- | A queue carrying tuples. The underlying structure is more
-- versatile and can be made to support arbitrary tuples.
data Queue2 a b = Queue2 { handle :: Handle }

type Handle = Tensor Ref ByteString

-- | Adds the given values to the queue.
enqueue :: forall a b v1 v2. (TensorType a, TensorType b)
           => Queue2 a b
           -> Tensor v1 a
           -> Tensor v2 b
           -> Build ControlNode
enqueue q =
    buildOp (opDef "QueueEnqueue"
             & opAttr "Tcomponents" .~ [ tensorType (undefined :: a)
                                       , tensorType (undefined :: b)])
    (handle q)

-- | Retrieves the values from the queue.
dequeue :: forall a b . (TensorType a, TensorType b)
           => Queue2 a b
           -> Build (Tensor Ref a, Tensor Ref b)
           -- ^ Dequeued tensors. They are paired in a sense
           -- that values appear together, even if they are
           -- not consumed together.
dequeue q =
    buildOp (opDef "QueueDequeue"
             & opAttr "component_types" .~ [ tensorType (undefined :: a)
                                           , tensorType (undefined :: b)])
    (handle q)

-- | Creates a new queue with the given capacity and shared name.
makeQueue2 :: forall a b . (TensorType a, TensorType b)
              => Int64  -- ^ The upper bound on the number of elements in
                        --  this queue. Negative numbers mean no limit.
              -> ByteString -- ^ If non-empty, this queue will be shared
                            -- under the given name across multiple sessions.
              -> Build (Queue2 a b)
makeQueue2 capacity sharedName = do
    q <- buildOp (opDef "FIFOQueue"
                     & opAttr "component_types" .~ [ tensorType (undefined :: a)
                                                   , tensorType (undefined :: b)]
                     & opAttr "shared_name" .~ sharedName
                     & opAttr "capacity" .~ capacity
                    )
    group q >>= addInitializer
    return (Queue2 q)

-- TODO(gnezdo): Figure out the closing story for queues.
