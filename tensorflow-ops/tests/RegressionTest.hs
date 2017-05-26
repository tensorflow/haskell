-- | Simple linear regression example for the README.

import Control.Monad (replicateM, replicateM_)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Variable as TF

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- replicateM 100 randomIO
    let yData = [x*3 + 8 | x <- xData]
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)

fit :: [Float] -> [Float] -> IO (Float, Float)
fit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')
