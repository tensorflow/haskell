-- | An implementation of ResourceHandle-based variables.
--
-- The main difference between this and 'Ref'-based variables is
-- that reads are explicit, via the 'readValue' op.
--
-- TODO: given that distinction, figure out a good story around
-- gradients and save/restore.  Then, merge this module into
-- TensorFlow.Ops.
{-# LANGUAGE RecursiveDo #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
module TensorFlow.Variable
    ( Variable
    , variable
    , variable'
    , readValue
    , initializedVariable
    , initializedVariable'
    , zeroInitializedVariable
    , zeroInitializedVariable'
    , assign
    , assign'
    , assignAdd
    , assignAdd'
    ) where

import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 ((.~), (&))
import TensorFlow.Core
import TensorFlow.Build (opDef)
import TensorFlow.BuildOp (buildInputs, pureOp, OpParams)
import TensorFlow.Output (opInputs, unNodeName)
import TensorFlow.Tensor (tensorNodeName)
import TensorFlow.Types (tensorType)
import qualified TensorFlow.GenOps.Core as CoreOps
import TensorFlow.Ops (zeros)

newtype Variable a = Variable (Tensor Value ResourceHandle)

-- | Creates a new, uninitialized variable.
variable :: (MonadBuild m, TensorType a) => Shape -> m (Variable a)
variable = variable' id

variable' :: forall m a . (MonadBuild m, TensorType a)
                    => OpParams -> Shape -> m (Variable a)
variable' params s = build $ do
    -- Each variable needs a unique "shared_name".  Use MonadFix to
    -- set the attribute to the same name as the variable itself, without
    -- exposing more internals of the Build module.
    rec t <- CoreOps.varHandleOp' (params . (opAttr "shared_name" .~ n))
                                    (tensorType (undefined :: a)) s
        let n = encodeUtf8 $ unNodeName $ tensorNodeName t
    return $ Variable t

-- | Creates a variable initialized to the given value.
-- Initialization happens next time session runs.
initializedVariable :: (MonadBuild m, TensorType a)
                    => Tensor v a -> m (Variable a)
initializedVariable = initializedVariable' id

initializedVariable' :: forall a m v . (MonadBuild m, TensorType a)
                    => OpParams -> Tensor v a -> m (Variable a)
initializedVariable' params initializer = do
    -- The shape is not known initially.
    v@(Variable h) <- variable' params (Shape [])
    i <- CoreOps.assignVariableOp h initializer
    addInitializer =<< group i
    return v

-- | Creates a zero-initialized variable with the given shape.
zeroInitializedVariable
  :: (MonadBuild m, TensorType a, Num a) => Shape -> m (Variable a)
zeroInitializedVariable = zeroInitializedVariable' id

zeroInitializedVariable'
  :: (MonadBuild m, TensorType a, Num a) => OpParams -> Shape -> m (Variable a)
zeroInitializedVariable' params = initializedVariable' params . zeros

-- | Gets the value stored in a variable.
--
-- Note that this op is stateful since it depends on the value of the variable;
-- however, it may be CSE'd with other reads in the same context.  The context can
-- be fixed by using 'render' along with (for example) 'withControlDependencies'.
-- For example:
--
-- >   runSession $ do
-- >     v <- variable []
-- >     a <- assign v 24
-- >     r <- withControlDependencies a $ render $ readValue v + 18
-- >     result <- run r
-- >     liftIO $ (42 :: Float) @=? unScalar result
--
--
readValue :: TensorType a => Variable a -> Tensor Build a
readValue = readValue' id

readValue' :: forall a . TensorType a
    => OpParams -> Variable a -> Tensor Build a
readValue' params (Variable h)
    = pureOp [] $ do
        os <- buildInputs h
        pure $ opDef "ReadVariableOp"
                & (params
                    . (opAttr "dtype" .~ tensorType (undefined :: a))
                    . (opInputs .~ os))

-- | Sets the value of a variable.
assign :: (MonadBuild m, TensorType a)
    => Variable a -> Tensor v a -> m ControlNode
assign = assign' id

assign' :: (MonadBuild m, TensorType a)
    => OpParams -> Variable a -> Tensor v a -> m ControlNode
assign' params (Variable h) v = CoreOps.assignVariableOp' params h v

-- | Increments the value of a variable.
assignAdd :: (MonadBuild m, TensorType a)
    => Variable a -> Tensor v a -> m ControlNode
assignAdd = assignAdd' id

assignAdd' :: (MonadBuild m, TensorType a)
    => OpParams -> Variable a -> Tensor v a -> m ControlNode
assignAdd' params (Variable h) v = CoreOps.assignAddVariableOp' params h v
