[![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-haskell-master)](https://ci.tensorflow.org/job/tensorflow-haskell-master)

The tensorflow-haskell package provides Haskell bindings to
[TensorFlow](https://www.tensorflow.org/).

This is not an official Google product.

# Documentation

https://tensorflow.github.io/haskell/haddock/

[TensorFlow.Core](https://tensorflow.github.io/haskell/haddock/tensorflow-0.1.0.0/TensorFlow-Core.html)
is a good place to start.

# Examples

Neural network model for the MNIST dataset: [code](tensorflow-mnist/app/Main.hs)

Toy example of a linear regression model
([full code](tensorflow-ops/tests/RegressionTest.hs)):

```haskell
import Control.Monad (replicateM, replicateM_, zipWithM)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF

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
    let yHat = (x `TF.mul` w) `TF.add` b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- gradientDescent 0.001 loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (w, b)
    return (w', b')

gradientDescent :: Float
                -> TF.Tensor TF.Value Float
                -> [TF.Tensor TF.Ref Float]
                -> TF.Session TF.ControlNode
gradientDescent alpha loss params = do
    let applyGrad param grad =
            TF.assign param (param `TF.sub` (TF.scalar alpha `TF.mul` grad))
    TF.group =<< zipWithM applyGrad params =<< TF.gradients loss params
```

# Installation Instructions

## Build with Docker on Linux

As an expedient we use [docker](https://www.docker.com/) for building. Once you have docker
working, the following commands will compile and run the tests.

    git clone --recursive https://github.com/tensorflow/haskell.git tensorflow-haskell
    cd tensorflow-haskell
    IMAGE_NAME=tensorflow/haskell:v0
    docker build -t $IMAGE_NAME docker
    # TODO: move the setup step to the docker script.
    stack --docker --docker-image=$IMAGE_NAME setup
    stack --docker --docker-image=$IMAGE_NAME test

There is also a demo application:

    cd tensorflow-mnist
    stack --docker --docker-image=$IMAGE_NAME build --exec Main

## Build on Mac OS X

Run the [install_osx_dependencies.sh](./tools/install_osx_dependencies.sh)
script in the `tools/` directory. The script installs dependencies
via [Homebrew](http://brew.sh) and then downloads and installs the TensorFlow
library on your machine under `/usr/local`.

After running the script to install system dependencies, build the project with stack: 

    stack test
