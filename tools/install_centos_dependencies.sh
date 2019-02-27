#!/usr/bin/env bash

echo "Installing CentOS 7.6 Dependencies"
echo "=================================="

# Basic environment
echo "Checking basic tools..."
sudo yum install wget git -y

# Protocol Buffers
echo "Preparing compiling environment..."
sudo yum install autoconf automake libtool unzip gcc-c++ -y
TMP_DIR=$(mktemp -d)
pushd $TMP_DIR
echo "Downloading protobuf..." 
git clone https://github.com/google/protobuf.git
sudo rm -rf /usr/local/src/protobuf
sudo mv protobuf /usr/local/src/
pushd /usr/local/src/protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=/usr
make
sudo make install
popd
popd

# snappy
echo "Installing snappy..."
sudo yum install snappy -y

# libtensorflow
TMP_DIR=$(mktemp -d)
pushd $TMP_DIR
echo "Downloading libtensorflow..."
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.9.0.tar.gz
echo "Installing libtensorflow..."
tar zxf libtensorflow-cpu-linux-x86_64-1.9.0.tar.gz
sudo cp -r lib/* /usr/lib64/
sudo cp -r include/* /usr/include/
sudo ldconfig
popd

echo ""
echo "Finished"
