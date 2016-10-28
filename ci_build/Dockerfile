# Image for launching TensorFlow Haskell tests.
FROM tensorflow/tensorflow:nightly-devel
# gcr.io/tensorflow/tensorflow:latest-devel
MAINTAINER TensorFlow authors <tensorflow-haskell@googlegroups.com>

RUN \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 575159689BEFB442 && \
    echo 'deb http://download.fpcomplete.com/ubuntu trusty main'| tee /etc/apt/sources.list.d/fpco.list

RUN apt-get update

RUN apt-get install -y \
        # Avoids /usr/bin/ld: cannot find -ltinfo
        libncurses5-dev \
        # Makes stack viable in the container
        libgmp-dev \
        libcurl3-dev \
        stack

# The build context directory is the top of the tensorflow-haskell
# tree.
ADD . /tfhs

WORKDIR /tfhs

# Installs protoc and the libraries.
RUN \
    cd third_party/tensorflow && \
    tensorflow/tools/ci_build/builds/configured CPU && \
    bazel --batch build -c opt '@protobuf//:protoc' && \
    install -s bazel-bin/external/protobuf/protoc /usr/local/bin && \
    bazel --batch build -c opt '//tensorflow:libtensorflow_c.so' && \
    install bazel-bin/tensorflow/libtensorflow_c.so /usr/local/lib && \
    bazel --batch clean

RUN stack setup && stack test --only-dependencies
