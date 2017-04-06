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

{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.ControlFlow
    ( -- * Dependencies
      withControlDependencies
    , group
      -- * Operations
    , noOp
    ) where

import TensorFlow.BuildOp
import TensorFlow.Build
import TensorFlow.Nodes

-- | Modify a 'Build' action, such that all new ops rendered in it will depend
-- on the nodes in the first argument.
withControlDependencies :: (MonadBuild m, Nodes t) => t -> m a -> m a
withControlDependencies deps act = do
    nodes <- build $ getNodes deps
    withNodeDependencies nodes act

-- TODO(judahjacobson): Reimplement withDependencies.

-- | Create an op that groups multiple operations.
--
-- When this op finishes, all ops in the input @n@ have finished.  This op has
-- no output.
group :: (MonadBuild m, Nodes t) => t -> m ControlNode
group deps = withControlDependencies deps noOp

-- | Does nothing.  Only useful as a placeholder for control edges.
noOp :: MonadBuild m => m ControlNode
noOp = build $ buildOp [] $ opDef "NoOp"
