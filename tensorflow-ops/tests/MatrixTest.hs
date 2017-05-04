{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}

import Control.Monad.IO.Class (liftIO)
import Control.Monad (replicateM_)

import qualified Data.Vector as V
import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (square)
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Variable as TF

import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import TensorFlow.Test (assertAllClose)

randomParam :: TF.Shape -> TF.Session (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape) = TF.truncatedNormal (TF.vector shape)

fitMatrix :: Test
fitMatrix = testCase "fitMatrix" $ TF.runSession $ do
  u <- TF.initializedVariable =<< randomParam [2, 1]
  v <- TF.initializedVariable =<< randomParam [1, 2]
  let ones = [1, 1, 1, 1] :: [Float]
      matx = TF.constant [2, 2] ones
      diff = matx `TF.sub` (TF.readValue u `TF.matMul` TF.readValue v)
      loss = TF.reduceMean $ TF.square diff
  trainStep <- TF.minimizeWith (TF.gradientDescent 0.01) loss [u, v]
  replicateM_ 1000 (TF.run trainStep)
  (u',v') <- TF.run (TF.readValue u, TF.readValue v)
  -- ones = u * v
  liftIO $ assertAllClose (V.fromList ones) ((*) <$> u' <*> v')

main :: IO ()
main = defaultMain [ fitMatrix ]
