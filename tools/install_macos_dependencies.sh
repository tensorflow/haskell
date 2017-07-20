#!/bin/bash
set -eu

echo "Installing macOS System Dependencies"
echo "=================================="

if ! type "brew" > /dev/null; then
    echo "Requires homebrew to be installed."
    echo "Install homebrew from https://brew.sh/"
    exit 1
fi

if brew ls --versions protobuf > /dev/null; then
    echo "protobuf installation detected."
else
    echo "protobuf not installed, installing with homebrew."
    brew install protobuf
fi

if brew ls --versions snappy > /dev/null; then
    echo "snappy installation detected."
else
    echo "snappy not installed, installing with homebrew."
    brew install snappy
fi

echo "Downloading libtensorflow..."
curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.0.0.tar.gz > libtensorflow.tar.gz

echo "Extracting and copying libtensorflow..."
sudo tar zxf libtensorflow.tar.gz -C /usr/local
rm libtensorflow.tar.gz
sudo mv /usr/local/lib/libtensorflow.so /usr/local/lib/libtensorflow.dylib

sudo install_name_tool -id libtensorflow.dylib /usr/local/lib/libtensorflow.dylib

echo "Installing submodule dependencies"
git submodule update --init --recursive

echo ""
echo "Finished"
