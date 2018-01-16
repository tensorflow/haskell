#!/bin/bash

# Builds a test image and runs the tests inside.

set -eux -o pipefail

IMAGE_NAME=tensorflow/haskell/ci_build:lts8

# Make sure we are in the root directory of the repositiory.
cd "$( dirname "$0" )"/..

git submodule update --init --recursive
docker build -t $IMAGE_NAME -f ci_build/Dockerfile .
docker run $IMAGE_NAME stack build --pedantic --test
