#!/bin/bash

# Builds a test image and runs the tests inside.

set -eu -o pipefail

IMAGE_NAME=tensorflow/haskell/ci_build:v0

git submodule update
docker build -t $IMAGE_NAME -f ci_build/Dockerfile .
docker run -ti  $IMAGE_NAME stack test
