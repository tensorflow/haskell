#!/bin/bash

# Builds a test image and runs the tests inside.

set -eu -o pipefail

STACK_RESOLVER=${STACK_RESOLVER:-lts-6.2}
IMAGE_NAME=tensorflow/haskell/ci_build:$STACK_RESOLVER

git submodule update
docker pull tensorflow/tensorflow:nightly-devel
docker build --build-arg STACK_RESOLVER=$STACK_RESOLVER -t $IMAGE_NAME -f ci_build/Dockerfile .
docker run $IMAGE_NAME stack build --resolver=$STACK_RESOLVER --pedantic --test
