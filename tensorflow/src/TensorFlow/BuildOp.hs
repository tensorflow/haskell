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

{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}

module TensorFlow.BuildOp
    ( OpResult
    , BuildOp
    , buildOp
    , buildListOp
    , eqLengthGuard
    )
  where

import Control.Monad (replicateM)
import Control.Monad.Reader (ReaderT, runReaderT, ask)
import Control.Monad.State.Strict (State, runState, get, put)
import Data.Int (Int64)
import Lens.Family2 ((&), (<>~), (^.))

import TensorFlow.Build
import TensorFlow.Output
import TensorFlow.Tensor

data ResultState = ResultState !OutputIx [Int64] deriving Show

type Result = ReaderT Op (State ResultState)

-- | Class of types that can be used as op outputs.
class OpResult a where
    toResult :: Result a

instance (OpResult a1, OpResult a2) => OpResult (a1, a2) where
    toResult = (,) <$> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3) => OpResult (a1, a2, a3) where
    toResult = (,,) <$> toResult <*> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3, OpResult a4)
         => OpResult (a1, a2, a3, a4) where
    toResult = (,,,) <$> toResult <*> toResult <*> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3, OpResult a4, OpResult a5)
         => OpResult (a1, a2, a3, a4, a5) where
    toResult = (,,,,) <$> toResult
                      <*> toResult
                      <*> toResult
                      <*> toResult
                      <*> toResult

instance ( OpResult a1
         , OpResult a2
         , OpResult a3
         , OpResult a4
         , OpResult a5
         , OpResult a6
         )
         => OpResult (a1, a2, a3, a4, a5, a6) where
    toResult = (,,,,,)
               <$> toResult
               <*> toResult
               <*> toResult
               <*> toResult
               <*> toResult
               <*> toResult

tensorResult :: TensorKind v -> Result (Tensor v a)
tensorResult v = Tensor v <$> recordResult

recordResult :: Result Output
recordResult = do
    o <- ask
    ResultState i ns <- get
    put $! ResultState (i+1) ns
    return $! output i o

instance OpResult (ResourceHandle a) where
    toResult = ResourceHandle <$> recordResult

instance OpResult (Tensor Value a) where
    toResult = tensorResult ValueKind

instance OpResult (Tensor Ref a) where
    toResult = tensorResult RefKind

instance OpResult ControlNode where
    toResult = ControlNode <$> ask

instance OpResult a => OpResult [a] where
    toResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in toResult. " ++
                          "Likely misuse of buildListOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) toResult

runResult :: OpResult a => [Int64] -> Op -> a
runResult ns o =
    case runState (runReaderT toResult o) (ResultState 0 ns) of
        (x, ResultState _ []) -> x
        (_, ns') -> error $ "Ununsed length in runResult attributes: " ++
                            show (ns, ns')

-- | Make a new "pure" op, which may be deduped with identical ops within
-- the same scope.
pureResult :: OpResult a => [Int64] -> OpDef -> [Output] -> a
pureResult ns o ts = runResult ns $ Unrendered $ addReversedInputs o ts

-- | Make a new "stateful" op, which will not be deduped with otherwise
-- identical ops.
buildResult :: OpResult a => [Int64] -> OpDef -> [Output] -> Build a
buildResult ns o ts
    = runResult ns . Rendered <$> addNewOp (addReversedInputs o ts)

addReversedInputs :: OpDef -> [Output] -> OpDef
addReversedInputs o ts = o & opInputs <>~ reverse ts

-- | Class of types that can be used as op functions.
class BuildOp f where
    buildOp' :: [Int64]  -- ^ Sizes of list results (having number_attr)
             -> OpDef
             -> [Output] -- ^ Accumulator for inputs to the op.
             -> f

-- | Starts an operation that returns a structured set of tensors
-- (singletons or tuples).
buildOp :: BuildOp f => OpDef -> f
buildOp o = buildOp' [] o []

-- | Starts an operation that returns a list of tensors.
buildListOp :: BuildOp f => [Int64]
               -- ^ Cardinality of the corresponding list of tensors output.
               -> OpDef -> f
buildListOp counts o = buildOp' counts o []

instance BuildOp ControlNode where
    buildOp' _ o ts = ControlNode $ Unrendered $ addReversedInputs o ts

instance BuildOp (ResourceHandle a) where
    buildOp' = pureResult

instance BuildOp (Tensor Value a) where
    buildOp' = pureResult

instance BuildOp (Tensor Ref a) where
    buildOp' = pureResult

instance BuildOp [Tensor Value a] where
    buildOp' = pureResult

instance (OpResult t1, OpResult t2) => BuildOp (t1, t2) where
    buildOp' = pureResult

instance (OpResult t1, OpResult t2, OpResult t3) => BuildOp (t1, t2, t3) where
    buildOp' = pureResult

instance (OpResult t1, OpResult t2, OpResult t3, OpResult t4)
         => BuildOp (t1, t2, t3, t4) where
    buildOp' = pureResult

instance (OpResult t1, OpResult t2, OpResult t3, OpResult t4, OpResult t5)
         => BuildOp (t1, t2, t3, t4, t5) where
    buildOp' = pureResult

instance ( OpResult t1
         , OpResult t2
         , OpResult t3
         , OpResult t4
         , OpResult t5
         , OpResult t6
         )
         => BuildOp (t1, t2, t3, t4, t5, t6) where
    buildOp' = pureResult

instance OpResult a => BuildOp (Build a) where
    buildOp' = buildResult

instance BuildOp f => BuildOp (ResourceHandle a -> f) where
    buildOp' rf o ts (ResourceHandle t) = buildOp' rf o (t : ts)

instance BuildOp f => BuildOp (Tensor v a -> f) where
    buildOp' rf o ts t = buildOp' rf o (t ^. tensorOutput : ts)

instance BuildOp f => BuildOp ([Tensor v a] -> f) where
    buildOp' rf o accum ts
        = buildOp' rf o (reverse (fmap (^. tensorOutput) ts) ++ accum)

-- | Returns true if all the integers in each tuple are identical.
-- Throws an error with a descriptive message if not.
eqLengthGuard :: [(String, [(String, Int)])] -> Bool
eqLengthGuard = all eachOk
  where
    eachOk (_, []) = True
    -- The next line has (== 1) . length . nub in disguise
    eachOk (numberAttrName, pairs@((_, x) : zs)) = all (\z -> snd z == x) zs ||
        error ("number_attr " ++ numberAttrName ++
               " contains tensors with different length " ++ show pairs)
