The tensorflow-haskell package provides Haskell bindings to
[TensorFlow](https://www.tensorflow.org/).

This is not an official Google product.

# Instructions

## Build

For now [docker](https://www.docker.com/) is required. Once you have docker
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
