![Build Status](https://storage.googleapis.com/tensorflow-haskell-kokoro-build-badges/github.png)

The tensorflow-haskell package provides Haskell bindings to
[TensorFlow](https://www.tensorflow.org/).

This is not an official Google product.

# Documentation

https://tensorflow.github.io/haskell/haddock/

[TensorFlow.Core](https://tensorflow.github.io/haskell/haddock/tensorflow/TensorFlow-Core.html)
is a good place to start.

# Examples

Neural network model for the MNIST dataset: [code](tensorflow-mnist/app/Main.hs)

Toy example of a linear regression model
([full code](tensorflow-ops/tests/RegressionTest.hs)):

```haskell
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
```

# Installation Instructions

Note: building this repository with `stack` requires version `1.4.0` or newer.
Check your stack version with `stack --version` in a terminal.

## Build with Docker on Linux

As an expedient we use [docker](https://www.docker.com/) for building. Once you have docker
working, the following commands will compile and run the tests.

```
git clone --recursive https://github.com/tensorflow/haskell.git tensorflow-haskell
cd tensorflow-haskell
IMAGE_NAME=tensorflow/haskell:v0
docker build -t $IMAGE_NAME docker
# TODO: move the setup step to the docker script.
stack --docker --docker-image=$IMAGE_NAME setup
stack --docker --docker-image=$IMAGE_NAME test
```

There is also a demo application:

```
cd tensorflow-mnist
stack --docker --docker-image=$IMAGE_NAME build --exec Main
```

### Stack + Docker + GPU

If you want to use GPU you can do:

```
IMAGE_NAME=tensorflow/haskell:1.14.0-gpu
docker build -t $IMAGE_NAME docker/gpu
```

### Using nvidia-docker version 2
See [Nvidia docker 2 install instructions](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

```
stack --docker --docker-image=$IMAGE_NAME setup
stack --docker --docker-run-args "--runtime=nvidia" --docker-image=$IMAGE_NAME test
```

### Using nvidia-docker classic

Stack needs to use `nvidia-docker` instead of the normal `docker` for GPU support. We must wrap 'docker' with a script. This script will shadow the normal `docker` command.

```
ln -s `pwd`/tools/nvidia-docker-wrapper.sh <somewhere in your path>/docker
stack --docker --docker-image=$IMAGE_NAME setup
stack --docker --docker-image=$IMAGE_NAME test
```

## Build on macOS

Run the [install_macos_dependencies.sh](./tools/install_macos_dependencies.sh)
script in the `tools/` directory. The script installs dependencies
via [Homebrew](https://brew.sh/) and then downloads and installs the TensorFlow
library on your machine under `/usr/local`.

After running the script to install system dependencies, build the project with stack:

    stack test

## Build on NixOS

The `shell.nix` provides an environment containing the necessary
dependencies. To build, run:

    $ stack --nix build

or alternatively you can run

    $ nix-shell

to enter the environment and build the project. Note, that it is an emulation
of common Linux environment rather than full-featured Nix package expression.
No exportable Nix package will appear, but local development is possible.

## Installation on CentOS

[Xiaokui Shu (@subbyte)](https://github.com/subbyte) maintains [separate instructions for installation on CentOS](https://github.com/subbyte/haskell-learn/blob/master/tensorflow_setup.md).

# Related Projects

## Statically validated tensor shapes

https://github.com/helq/tensorflow-haskell-deptyped is experimenting with using dependent types to statically validate tensor shapes. May be merged with this repository in the future.

Example:

```haskell
{-# LANGUAGE DataKinds, ScopedTypeVariables #-}

import Data.Maybe (fromJust)
import Data.Vector.Sized (Vector, fromList)
import TensorFlow.DepTyped

test :: IO (Vector 8 Float)
test = runSession $ do
  (x :: Placeholder "x" '[4,3] Float) <- placeholder

  let elems1 = fromJust $ fromList [1,2,3,4,1,2]
      elems2 = fromJust $ fromList [5,6,7,8]
      (w :: Tensor '[3,2] '[] Build Float) = constant elems1
      (b :: Tensor '[4,1] '[] Build Float) = constant elems2
      y = (x `matMul` w) `add` b -- y shape: [4,2] (b shape is [4.1] but `add` broadcasts it to [4,2])

  let (inputX :: TensorData "x" [4,3] Float) =
          encodeTensorData . fromJust $ fromList [1,2,3,4,1,0,7,9,5,3,5,4]

  runWithFeeds (feed x inputX :~~ NilFeedList) y

main :: IO ()
main = test >>= print
```

# License
This project is licensed under the terms of the [Apache 2.0 license](LICENSE).
