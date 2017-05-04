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

-- | This module contains definitions for some built-in TensorFlow operations.
--
-- Note that certain, "stateful" ops like 'variable' and 'assign' return a
-- 'Build' action (e.g., @Build (Tensor Ref a)@ instead of a pure value; the
-- returned 'Tensor's are always rendered in the current 'Build' context.  This
-- approach helps us avoid problems with inlining or common subexpression
-- elimination, by writing
--
-- > do
-- >     v <- variable []
-- >     w <- assign v 3
-- >     render $ w * w
--
-- instead of
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- > in w * w
--
-- since the latter could be reasonably transformed by the compiler into (or
-- vice versa)
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- >    w' = assign v 3
-- > in w * w'
--
-- Ops should return a 'Build' action if their original 'OpDef' marks them as
-- stateful, or if they take any Refs as input.  (This mirrors the rules that
-- TensorFlow uses to avoid common subexpression elimination.)
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module TensorFlow.Ops
    ( CoreOps.add
    , CoreOps.add'
    , CoreOps.abs
    , CoreOps.abs'
    , CoreOps.addN
    , CoreOps.addN'
    , CoreOps.argMax
    , CoreOps.argMax'
    , CoreOps.assign
    , CoreOps.assign'
    , CoreOps.broadcastGradientArgs
    , CoreOps.broadcastGradientArgs'
    , CoreOps.cast
    , CoreOps.cast'
    , CoreOps.concat
    , CoreOps.concat'
    , constant
    , constant'
    , CoreOps.equal
    , CoreOps.equal'
    , expandDims
    , expandDims'
    , initializedVariable
    , initializedVariable'
    , zeroInitializedVariable
    , zeroInitializedVariable'
    , CoreOps.fill
    , CoreOps.fill'
    , CoreOps.identity
    , CoreOps.identity'
    , CoreOps.matMul
    , CoreOps.matMul'
    , matTranspose
    , matTranspose'
    , CoreOps.mean
    , CoreOps.mean'
    , CoreOps.mul
    , CoreOps.mul'
    , CoreOps.neg
    , CoreOps.neg'
    , CoreOps.oneHot
    , CoreOps.oneHot'
    , CoreOps.pack
    , CoreOps.pack'
    , placeholder
    , placeholder'
    , CoreOps.range
    , CoreOps.range'
    , reducedShape
    , reduceMean
    , reduceMean'
    , CoreOps.relu
    , CoreOps.relu'
    , CoreOps.reluGrad
    , CoreOps.reluGrad'
    , CoreOps.reshape
    , CoreOps.reshape'
    , restore
    , restoreFromName
    , save
    , scalar
    , scalar'
    , shape
    , shape'
    , CoreOps.sign
    , CoreOps.sign'
    , CoreOps.size
    , CoreOps.size'
    , CoreOps.softmax
    , CoreOps.softmax'
    , CoreOps.softmaxCrossEntropyWithLogits
    , CoreOps.softmaxCrossEntropyWithLogits'
    , CoreOps.sparseToDense
    , CoreOps.sparseToDense'
    , CoreOps.sub
    , CoreOps.sub'
    , CoreOps.sum
    , CoreOps.sum'
    , reduceSum
    , reduceSum'
    , CoreOps.transpose
    , CoreOps.transpose'
    , truncatedNormal
    , truncatedNormal'
    , CoreOps.variable
    , CoreOps.variable'
    , vector
    , vector'
    , zeros
    , CoreOps.zerosLike
    , CoreOps.zerosLike'
    , scalarize
    ) where

import Data.ByteString (ByteString)
import Data.Complex (Complex)
import Data.Int (Int32, Int64)
import Data.Word (Word16)
import Prelude hiding (abs, sum, concat)
import Data.ProtoLens (def)
import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 ((.~), (&))
import Text.Printf (printf)
import Proto.Tensorflow.Core.Framework.Tensor
    ( TensorProto
    , dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape
  as TensorShape
import TensorFlow.Build
import TensorFlow.BuildOp
import TensorFlow.ControlFlow (group)
import TensorFlow.Tensor
import TensorFlow.Types

import qualified TensorFlow.GenOps.Core as CoreOps

import qualified Prelude (abs)

-- TODO: Look into hs-boot refactoring to allow mutually recursive imports.
-- | Must be defined as an orphan because of the dependency order between Ops
-- and Tensor.
--
-- The indirect constraint "v ~ Value" helps disambiguate types, for example in
-- "neg 1 :: Tensor Value Float", it helps find the type of the subexpression
-- "1".
instance ( TensorType a
         , Num a
         , v ~ Build
         , OneOf '[ Double, Float, Int32, Int64
                  , Complex Float, Complex Double] a) => Num (Tensor v a) where
    (+) = CoreOps.add
    (*) = CoreOps.mul
    (-) = CoreOps.sub
    abs = CoreOps.abs
    fromInteger = scalar . fromInteger
    signum = CoreOps.sign
    negate = CoreOps.neg

matTranspose :: TensorType a => Tensor e a -> Tensor Build a
matTranspose = matTranspose' id

matTranspose' :: TensorType a => OpParams -> Tensor v a -> Tensor Build a
matTranspose' params = flip (CoreOps.transpose' params) (vector [1, 0 :: Int32])

placeholder :: (MonadBuild m, TensorType a) => Shape -> m (Tensor Value a)
placeholder = placeholder' id

placeholder' :: forall m a . (MonadBuild m, TensorType a)
             => OpParams -> Shape -> m (Tensor Value a)
placeholder' params pShape
    -- Note: we don't use CoreOps.placeholder' since that op isn't stateful,
    -- and thus would be CSE'd.
    = build $ buildOp [] $ opDef "Placeholder"
                & opAttr "dtype" .~ tensorType (undefined :: a)
                & opAttr "shape" .~ pShape
                & params

-- | Creates a variable initialized to the given value.
-- Initialization happens next time session runs.
initializedVariable :: (MonadBuild m, TensorType a)
                    => Tensor v a -> m (Tensor Ref a)
initializedVariable = initializedVariable' id

initializedVariable' :: (MonadBuild m, TensorType a)
                    => OpParams -> Tensor v a -> m (Tensor Ref a)
initializedVariable' params initializer = do
    v <- CoreOps.variable' params []  -- The shape is not known initially.
    i <- CoreOps.assign' (opAttr "validate_shape" .~ False) v
                            initializer
    addInitializer =<< group i
    return v

-- | Creates a zero-initialized variable with the given shape.
zeroInitializedVariable
  :: (MonadBuild m, TensorType a, Num a) =>
     TensorFlow.Types.Shape -> m (Tensor TensorFlow.Tensor.Ref a)
zeroInitializedVariable = zeroInitializedVariable' id

zeroInitializedVariable'
  :: (MonadBuild m, TensorType a, Num a) =>
     OpParams -> TensorFlow.Types.Shape -> m (Tensor TensorFlow.Tensor.Ref a)
zeroInitializedVariable' params = initializedVariable' params . zeros

-- TODO: Support heterogeneous list of tensors.
save :: forall a m v . (Rendered (Tensor v), MonadBuild m, TensorType a)
        => ByteString    -- ^ File path.
        -> [Tensor v a]  -- ^ Tensors to save.
        -> m ControlNode
save path xs = build $ do
    let toByteStringTensor = scalar . encodeUtf8 . encodeOutput . renderedOutput
    let names = fmap toByteStringTensor xs
    let types = replicate (length xs) (tensorType (undefined :: a))
    names' <- buildInputs $ CoreOps.pack names
    xs' <- buildInputs xs
    path' <- buildInputs $ scalar path
    buildOp [] $ opDef "Save"
                    & opAttr "T" .~ types
                    & opInputs .~ (path' ++ names' ++ xs')

-- | Restore a tensor's value from a checkpoint file.
--
-- This version allows restoring from a checkpoint file that uses a different
-- tensor name than the variable.
restoreFromName :: forall a m . (MonadBuild m, TensorType a)
                => ByteString    -- ^ File path.
                -> ByteString    -- ^ Tensor name override.
                -> Tensor Ref a  -- ^ Tensor to restore.
                -> m ControlNode
restoreFromName path name x = build $ do
    path' <- buildInputs $ scalar path
    name' <- buildInputs $ scalar name
    restoreOp <- buildOp [] $ opDef "Restore"
                               & opAttr "dt" .~ tensorType (undefined :: a)
                               & opInputs .~ (path' ++ name')
    group =<< CoreOps.assign x (restoreOp :: Tensor Value a)

-- | Restore a tensor's value from a checkpoint file.
restore :: forall a m . (MonadBuild m, TensorType a)
        => ByteString    -- ^ File path.
        -> Tensor Ref a  -- ^ Tensor to restore.
        -> m ControlNode
restore path x = restoreFromName path name x
  where
    name = encodeUtf8 $ encodeOutput $ renderedOutput x

-- | Create a constant tensor.
--
-- The values should be in row major order, e.g.,
--
--   element 0:   index (0, ..., 0)
--   element 1:   index (0, ..., 1)
--   ...
constant :: TensorType a => Shape -> [a] -> Tensor Build a
constant = constant' id

constant' :: forall a . TensorType a => OpParams -> Shape -> [a] -> Tensor Build a
constant' params (Shape cShape) values
    | invalidLength = error invalidLengthMsg
    | otherwise = CoreOps.const' (params . (opAttr "value" .~ typedNode))
  where
    invalidLength = product cShape /= fromIntegral (length values)
    invalidLengthMsg = printf "invalid tensor length: expected %d got %d"
                              (product cShape)
                              (length values)
    typedNode :: TensorProto
    typedNode = def
                & dtype .~ tensorType (undefined :: a)
                & tensorShape.TensorShape.dim .~
                      [def & TensorShape.size .~ x | x <- cShape]
                & tensorVal .~ values

-- | Reshape a N-D tensor down to a scalar.
--
-- See `TensorFlow.GenOps.Core.reshape`.
scalarize :: TensorType a => Tensor v a -> Tensor Build a
scalarize t = CoreOps.reshape t (vector scalarShape)
    where
        scalarShape = [] :: [Int32]

-- | Sum a tensor down to a scalar
-- Seee `TensorFlow.GenOps.Core.sum`
reduceSum :: (OneOf '[ Double, Float, Int32, Int64
                     , Complex Float, Complex Double] a) =>
             Tensor v a -> Tensor Build a
reduceSum x = CoreOps.sum x allAxes
  where allAxes = CoreOps.range 0 (CoreOps.rank x :: Tensor Build Int32) 1

reduceSum' :: (OneOf '[ Double, Float, Int32, Int64
                      , Complex Float, Complex Double] a) =>
              OpParams -> Tensor v a -> Tensor Build a
reduceSum' params x = CoreOps.sum' params x allAxes
  where allAxes = CoreOps.range 0 (CoreOps.rank x :: Tensor Build Int32) 1

-- | Computes the mean of elements across dimensions of a tensor.
-- See `TensorFlow.GenOps.Core.mean`
reduceMean
  :: ( TensorType a
     , OneOf '[ Double, Float, Complex Float, Complex Double] a
     )
  => Tensor v a -> Tensor Build a
reduceMean = reduceMean' id

reduceMean'
  :: ( TensorType a
     , OneOf '[ Double, Float, Complex Float, Complex Double] a
     )
  => OpParams -> Tensor v a -> Tensor Build a
reduceMean' params x = CoreOps.mean' params x allAxes
  where allAxes = CoreOps.range 0 (CoreOps.rank x :: Tensor Build Int32) 1

-- | Create a constant vector.
vector :: TensorType a => [a] -> Tensor Build a
vector = vector' id

vector' :: TensorType a => OpParams -> [a] -> Tensor Build a
vector' params xs = constant' params [fromIntegral $ length xs] xs

-- | Create a constant scalar.
scalar :: TensorType a => a -> Tensor Build a
scalar = scalar' id

scalar' :: TensorType a => OpParams -> a -> Tensor Build a
scalar' params x = constant' params [] [x]

-- | Random tensor from the unit normal distribution with bounded values.
--
-- This is a type-restricted version of 'TensorFlow.GenOps.Core.truncatedNormal'.
truncatedNormal :: (MonadBuild m, OneOf '[Word16, Double, Float] a)
                => Tensor v Int64  -- ^ Shape.
                -> m (Tensor Value a)
truncatedNormal = CoreOps.truncatedNormal

truncatedNormal' :: (MonadBuild m, OneOf '[Word16, Double, Float] a)
                => OpParams -> Tensor v Int64  -- ^ Shape.
                -> m (Tensor Value a)
truncatedNormal' = CoreOps.truncatedNormal'

zeros :: forall a . (Num a, TensorType a) => Shape -> Tensor Build a
zeros (Shape s) = CoreOps.fill (vector $ map fromIntegral s) (scalar 0)

shape :: TensorType t => Tensor v t -> Tensor Build Int32
shape = CoreOps.shape

shape' :: TensorType t => OpParams -> Tensor v t -> Tensor Build Int32
shape' = CoreOps.shape'

expandDims :: TensorType t => Tensor v1 t -> Tensor v2 Int32 -> Tensor Build t
expandDims = CoreOps.expandDims

expandDims' :: TensorType t => OpParams -> Tensor v1 t -> Tensor v2 Int32 -> Tensor Build t
expandDims' = CoreOps.expandDims'

-- | Helper function for reduction ops (translation of math_ops.reduced_shape).
reducedShape :: (OneOf '[ Int32, Int64 ] t1, OneOf '[ Int32, Int64 ] t2) =>
                Tensor v1 t1 -> Tensor v2 t2 -> Tensor Build Int32
reducedShape inputShape axes =
    let inputShape32 = toInt32 inputShape         -- [2, 3, 5, 7]
        axes32 = toInt32 axes                     -- [1, 2]
        toInt32 x = CoreOps.cast x :: Tensor Build Int32
        inputRank = CoreOps.size inputShape32     -- 4
        axesMod = (axes32 + inputRank) `CoreOps.mod` inputRank
        axesShape = shape axesMod                 -- [2]
    in CoreOps.dynamicStitch                      -- [2, 1, 1, 7]
         [CoreOps.range 0 inputRank 1,            -- [0, 1, 2, 3]
           axesMod]                               -- [1, 2]
         [inputShape32,                           -- [2, 3, 5, 7]
           CoreOps.fill axesShape 1]              -- [1, 1]
