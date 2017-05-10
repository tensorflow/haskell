#!/bin/bash

# Builds a test image and runs the tests inside.

set -eu -o pipefail

IMAGE_NAME=tensorflow/haskell/ci_build:lts8

git submodule update
docker build -t $IMAGE_NAME -f ci_build/Dockerfile .
docker run $IMAGE_NAME stack build --pedantic --test
