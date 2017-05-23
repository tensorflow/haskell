-- | An implementation of ResourceHandle-based variables.
--
-- The main difference between this and 'Ref'-based variables is
-- that reads are explicit, via the 'readValue' op.
--
-- TODO: given that distinction, figure out a good story around
-- gradients and save/restore.  Then, merge this module into
-- TensorFlow.Ops.
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecursiveDo #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
module TensorFlow.Variable
    ( Variable
    , variable
    , variable'
    , readValue
    , initializedValue
    , initializedVariable
    , initializedVariable'
    , zeroInitializedVariable
    , zeroInitializedVariable'
    , assign
    , assign'
    , assignAdd
    , assignAdd'
    , resourceApplyAdam
    , resourceApplyAdam'
    ) where

import qualified Data.Complex
import qualified Data.Int
import qualified Data.Word
import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 ((.~), (&))
import TensorFlow.Core
import TensorFlow.Build (opDef)
import TensorFlow.BuildOp (buildInputs, pureOp, OpParams)
import TensorFlow.Output (opInputs, unNodeName)
import TensorFlow.Tensor (Rendered(..), ToTensor(..), renderValue, tensorNodeName)
import TensorFlow.Types (tensorType)
import qualified TensorFlow.GenOps.Core as CoreOps
import TensorFlow.Ops (zeros)

data Variable a = Variable
    { variableHandle   :: Tensor Value ResourceHandle
    , initializedValue :: Maybe (Tensor Value a)
      -- ^ The initial value of a 'Variable' created with 'initializedVariable'.
    }

instance Rendered Variable where
    renderedOutput = renderedOutput . variableHandle

instance ToTensor Variable where
    toTensor = readValue

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
    return $ Variable t Nothing

-- | Creates a variable initialized to the given value.
-- Initialization happens next time session runs.
initializedVariable :: (MonadBuild m, TensorType a)
                    => Tensor v a -> m (Variable a)
initializedVariable = initializedVariable' id

initializedVariable' :: forall a m v . (MonadBuild m, TensorType a)
                    => OpParams -> Tensor v a -> m (Variable a)
initializedVariable' params initializer = do
    -- The shape is not known initially.
    (Variable h Nothing :: Variable a) <- variable' params (Shape [])
    initializer' <- renderValue initializer
    i <- CoreOps.assignVariableOp h initializer'
    addInitializer =<< group i
    return (Variable h (Just initializer'))

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
readValue' params (Variable h _)
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
assign' params (Variable h _) v = CoreOps.assignVariableOp' params h v

-- | Increments the value of a variable.
assignAdd :: (MonadBuild m, TensorType a)
    => Variable a -> Tensor v a -> m ControlNode
assignAdd = assignAdd' id

assignAdd' :: (MonadBuild m, TensorType a)
    => OpParams -> Variable a -> Tensor v a -> m ControlNode
assignAdd' params (Variable h _) v = CoreOps.assignAddVariableOp' params h v

-- | Update '*var' according to the Adam algorithm.
--
-- lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
-- m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
-- v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
-- variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
resourceApplyAdam ::
    (MonadBuild m,
     OneOf '[(Data.Complex.Complex Double),
             (Data.Complex.Complex Float),
             Data.Int.Int16,
             Data.Int.Int32,
             Data.Int.Int64, Data.Int.Int8,
             Data.Word.Word16,
             Data.Word.Word8, Double,
             Float] t)
    => Variable t -- ^ __var__: Should be from a Variable().
    -> Variable t -- ^ __m__: Should be from a Variable().
    -> Variable t -- ^ __v__: Should be from a Variable().
    -> Tensor v1 t -- ^ __beta1_power__: Must be a scalar.
    -> Tensor v2 t -- ^ __beta2_power__: Must be a scalar.
    -> Tensor v3 t -- ^ __lr__: Scaling factor. Must be a scalar.
    -> Tensor v4 t -- ^ __beta1__: Momentum factor. Must be a scalar.
    -> Tensor v5 t -- ^ __beta2__: Momentum factor. Must be a scalar.
    -> Tensor v6 t -- ^ __epsilon__: Ridge term. Must be a scalar.
    -> Tensor v7 t -- ^ __grad__: The gradient.
    -> m (ControlNode)
resourceApplyAdam = resourceApplyAdam' id

resourceApplyAdam' ::
    (MonadBuild m,
     OneOf '[(Data.Complex.Complex Double),
             (Data.Complex.Complex Float),
             Data.Int.Int16, Data.Int.Int32,
             Data.Int.Int64, Data.Int.Int8,
             Data.Word.Word16, Data.Word.Word8, Double,
             Float] t)
    => OpParams
    -> Variable t -- ^ __var__: Should be from a Variable().
    -> Variable t -- ^ __m__: Should be from a Variable().
    -> Variable t -- ^ __v__: Should be from a Variable().
    -> Tensor v1 t -- ^ __beta1_power__: Must be a scalar.
    -> Tensor v2 t -- ^ __beta2_power__: Must be a scalar.
    -> Tensor v3 t -- ^ __lr__: Scaling factor. Must be a scalar.
    -> Tensor v4 t -- ^ __beta1__: Momentum factor. Must be a scalar.
    -> Tensor v5 t -- ^ __beta2__: Momentum factor. Must be a scalar.
    -> Tensor v6 t -- ^ __epsilon__: Ridge term. Must be a scalar.
    -> Tensor v7 t -- ^ __grad__: The gradient.
    -> m (ControlNode)
resourceApplyAdam' params (Variable var _) (Variable m _) (Variable v _) =
    CoreOps.resourceApplyAdam' params var m v
