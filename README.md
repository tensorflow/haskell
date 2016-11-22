[![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-haskell-master)](https://ci.tensorflow.org/job/tensorflow-haskell-master)

The tensorflow-haskell package provides Haskell bindings to
[TensorFlow](https://www.tensorflow.org/).

This is not an official Google product.

# Instructions

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
