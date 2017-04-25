{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}

import Data.Int (Int32, Int64)

import Control.Monad.IO.Class (liftIO)
import Control.Monad (replicateM_, zipWithM)

import TensorFlow.GenOps.Core (square)
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF
import qualified Data.Vector as V

import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import TensorFlow.Test (assertAllClose)
import Google.Test (googleTest)


randomParam (TF.Shape shape) =
  (`TF.mul` (TF.scalar 1.0)) <$> TF.truncatedNormal (TF.vector shape)

reduceMean :: TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float
reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))

fitMatrix :: Test
fitMatrix = testCase "fitMatrix" $ TF.runSession $ do
  u <- TF.initializedVariable =<< randomParam [2, 1]
  v <- TF.initializedVariable =<< randomParam [1, 2]
  let ones = [1, 1, 1, 1] :: [Float]
      matx = TF.constant [2, 2] ones
      diff = matx `TF.sub` (u `TF.matMul` v)
      loss = reduceMean . reduceMean $ square diff
  trainStep <- gradientDescent 0.01 loss [u, v]
  replicateM_ 1000 (TF.run trainStep)
  (u',v') <- TF.run (u, v)
  -- ones = u * v
  liftIO $ assertAllClose (V.fromList ones) ((*) <$> u' <*> v')
  
gradientDescent :: Float
                -> TF.Tensor TF.Build Float
                -> [TF.Tensor TF.Ref Float]
                -> TF.Session TF.ControlNode
gradientDescent alpha loss params = do
    let applyGrad param grad =
            TF.assign param (param `TF.sub` (TF.scalar alpha `TF.mul` grad))
    TF.group =<< zipWithM applyGrad params =<< TF.gradients loss params

main :: IO ()
main = googleTest [ fitMatrix ]
