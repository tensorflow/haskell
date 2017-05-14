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

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module TensorFlow.BuildOp
    ( BuildResult(..)
    , buildOp
    , PureResult(..)
    , pureOp
    , eqLengthGuard
    , BuildInputs(..)
    , OpParams
    )
  where

import Control.Monad (liftM2, replicateM)
import Control.Monad.Reader (ReaderT, runReaderT, ask)
import Control.Monad.State.Strict (State, evalState, get, put)
import Data.Int (Int64)

import TensorFlow.Build
import TensorFlow.Output
import TensorFlow.Tensor
import TensorFlow.Types

data ResultState = ResultState !OutputIx [Int64] deriving Show

type Result = ReaderT NodeName (State ResultState)

-- | Class of types that can be used as op outputs.
class BuildResult a where
    buildResult :: Result a

instance (BuildResult a1, BuildResult a2) => BuildResult (a1, a2) where
    buildResult = (,) <$> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3) => BuildResult (a1, a2, a3) where
    buildResult = (,,) <$> buildResult <*> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3, BuildResult a4)
         => BuildResult (a1, a2, a3, a4) where
    buildResult = (,,,) <$> buildResult <*> buildResult <*> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3, BuildResult a4, BuildResult a5)
         => BuildResult (a1, a2, a3, a4, a5) where
    buildResult = (,,,,) <$> buildResult
                      <*> buildResult
                      <*> buildResult
                      <*> buildResult
                      <*> buildResult

instance ( BuildResult a1
         , BuildResult a2
         , BuildResult a3
         , BuildResult a4
         , BuildResult a5
         , BuildResult a6
         )
         => BuildResult (a1, a2, a3, a4, a5, a6) where
    buildResult = (,,,,,)
               <$> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult

instance ( BuildResult a1
         , BuildResult a2
         , BuildResult a3
         , BuildResult a4
         , BuildResult a5
         , BuildResult a6
         , BuildResult a7
         )
         => BuildResult (a1, a2, a3, a4, a5, a6, a7) where
    buildResult = (,,,,,,)
               <$> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult

instance ( BuildResult a1
         , BuildResult a2
         , BuildResult a3
         , BuildResult a4
         , BuildResult a5
         , BuildResult a6
         , BuildResult a7
         , BuildResult a8
         )
         => BuildResult (a1, a2, a3, a4, a5, a6, a7, a8) where
    buildResult = (,,,,,,,)
               <$> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult

recordResult :: Result Output
recordResult = do
    o <- ask
    ResultState i ns <- get
    put $! ResultState (i+1) ns
    return $! output i o

instance (TensorKind v, Rendered (Tensor v)) => BuildResult (Tensor v a) where
    buildResult = Tensor . pure <$> recordResult

instance BuildResult ControlNode where
    buildResult = ControlNode <$> ask

instance (TensorKind v, Rendered (Tensor v), TensorTypes as) => BuildResult (TensorList v as) where
  buildResult = loop (tensorTypes :: TensorTypeList as)
    where
        loop :: TensorTypeList bs -> Result (TensorList v bs)
        loop Nil = return Nil
        loop (TensorTypeProxy :/ ls) = do
            t <- buildResult
            ts <- loop ls
            return (t :/ ts)

instance BuildResult a => BuildResult [a] where
    buildResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in buildResult. " ++
                          "Likely misuse of buildOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) buildResult

buildOp :: BuildResult a => [Int64] -> OpDef -> Build a
buildOp sizes o = do
    n <- addNewOp o
    return $ flip evalState (ResultState 0 sizes) (runReaderT buildResult n)

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

-----------


-- | Class of types that can be used as op outputs.
class PureResult a where
    pureResult :: ReaderT (Build OpDef) (State ResultState) a

instance PureResult (Tensor Build a) where
    pureResult = do
        ResultState i ns <- get
        put $! ResultState (i+1) ns
        makeOp <- ask
        return $ Tensor $ do
            o <- makeOp
            -- TODO: unify with BuildResult (Tensor v)
            output i <$> getOrAddOp o

instance (PureResult a1, PureResult a2) => PureResult (a1, a2) where
    pureResult = (,) <$> pureResult <*> pureResult

instance (PureResult a1, PureResult a2, PureResult a3) => PureResult (a1, a2, a3) where
    pureResult = (,,) <$> pureResult <*> pureResult <*> pureResult

instance (PureResult a1, PureResult a2, PureResult a3, PureResult a4)
         => PureResult (a1, a2, a3, a4) where
    pureResult = (,,,) <$> pureResult <*> pureResult <*> pureResult <*> pureResult

instance (PureResult a1, PureResult a2, PureResult a3, PureResult a4, PureResult a5)
         => PureResult (a1, a2, a3, a4, a5) where
    pureResult = (,,,,) <$> pureResult
                      <*> pureResult
                      <*> pureResult
                      <*> pureResult
                      <*> pureResult

instance ( PureResult a1
         , PureResult a2
         , PureResult a3
         , PureResult a4
         , PureResult a5
         , PureResult a6
         )
         => PureResult (a1, a2, a3, a4, a5, a6) where
    pureResult = (,,,,,)
               <$> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult

instance ( PureResult a1
         , PureResult a2
         , PureResult a3
         , PureResult a4
         , PureResult a5
         , PureResult a6
         , PureResult a7
         )
         => PureResult (a1, a2, a3, a4, a5, a6, a7) where
    pureResult = (,,,,,,)
               <$> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult

instance ( PureResult a1
         , PureResult a2
         , PureResult a3
         , PureResult a4
         , PureResult a5
         , PureResult a6
         , PureResult a7
         , PureResult a8
         )
         => PureResult (a1, a2, a3, a4, a5, a6, a7, a8) where
    pureResult = (,,,,,,,)
               <$> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult
               <*> pureResult

instance PureResult a => PureResult [a] where
    pureResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in pureResult. " ++
                          "Likely misuse of pureOp with output lists."
            n : rest -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) pureResult

instance TensorTypes as => PureResult (TensorList Build as) where
    pureResult = loop (tensorTypes :: TensorTypeList as)
      where
        loop :: TensorTypeList bs -> ReaderT (Build OpDef) (State ResultState)
                                        (TensorList Build bs)
        loop Nil = return Nil
        loop (TensorTypeProxy :/ ls) = do
            t <- pureResult
            ts <- loop ls
            return (t :/ ts)

pureOp :: PureResult a => [Int64] -> Build OpDef -> a
pureOp sizes o = flip evalState (ResultState 0 sizes) (runReaderT pureResult o)

-----
-- Class of types that can be used as arguments

class BuildInputs a where
    buildInputs :: a -> Build [Output]

instance BuildInputs a => BuildInputs [a] where
    buildInputs = fmap concat . mapM buildInputs

instance BuildInputs (Tensor v a) where
    buildInputs (Tensor t) = do
        o <- toBuild t
        return [o]

instance BuildInputs (ListOf (Tensor v) as) where
    buildInputs Nil = return []
    buildInputs (t :/ ts) = liftM2 (++) (buildInputs t) (buildInputs ts)

----

-- | Parameters to build an op (for example, the node name or optional attributes).
-- TODO: be more type safe.
type OpParams = OpDef -> OpDef
