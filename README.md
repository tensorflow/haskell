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
    w <- TF.build (TF.initializedVariable 0)
    b <- TF.build (TF.initializedVariable 0)
    -- Define the loss function.
    let yHat = (x `TF.mul` w) `TF.add` b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.build (gradientDescent 0.001 loss [w, b])
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (w, b)
    return (w', b')

gradientDescent :: Float
                -> TF.Tensor TF.Value Float
                -> [TF.Tensor TF.Ref Float]
                -> TF.Build TF.ControlNode
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

The following instructions were verified with Mac OS X El Capitan.

- Install the "protoc" binary somewhere in your PATH. You can get it by
  downloading the corresponding file for your system from
  https://github.com/google/protobuf/releases. (The corresponding file will be
  named something like `protoc-*-.zip`.)

- Install dependencies via [Homebrew](http://brew.sh):

        brew install swig
        brew install bazel

- Build the TensorFlow library and install it on your machine:

        cd third_party/tensorflow
        ./configure  # Choose the defaults when prompted
        bazel build -c opt tensorflow:libtensorflow_c.so
        install bazel-bin/tensorflow/libtensorflow_c.so /usr/local/lib/libtensorflow_c.dylib
        install_name_tool -id libtensorflow_c.dylib /usr/local/lib/libtensorflow_c.dylib
        cd ../..

- Run stack:

        stack test

Note: you may need to upgrade your version of Clang if you get an error like the following:

    tensorflow/core/ops/ctc_ops.cc:60:7: error: return type 'tensorflow::Status' must match previous return type 'const ::tensorflow::Status' when lambda expression has unspecified explicit return type
        return Status::OK();

In that case you can just upgrade XCode and then run `gcc --version` to get the new version of the compiler.
