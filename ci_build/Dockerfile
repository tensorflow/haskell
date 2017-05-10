# Image for launching TensorFlow Haskell tests.  Unlike the
# development image in docker/Dockerfile, this one doesn't require
# stack to be installed on the host.  This comes at the expense of
# flexibility.

FROM tensorflow/tensorflow:1.0.0
MAINTAINER TensorFlow authors <tensorflow-haskell@googlegroups.com>

# The build context directory is the top of the tensorflow-haskell
# tree.
ADD . /tfhs
WORKDIR /tfhs

RUN \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 575159689BEFB442 && \
    echo 'deb http://download.fpcomplete.com/ubuntu trusty main'| tee /etc/apt/sources.list.d/fpco.list && \
    apt-get update && \
    apt-get install -y \
        # Required by snappy-frames dependency.
        libsnappy-dev \
        # Avoids /usr/bin/ld: cannot find -ltinfo
        libncurses5-dev \
        # Makes stack viable in the container
        libgmp-dev \
        libcurl3-dev \
        stack && \
    # Installs protoc and the libraries.
    curl -O -L https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip -d /usr/local protoc-3.2.0-linux-x86_64.zip bin/protoc && \
    chmod 755 /usr/local/bin/protoc && \
    curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.0.0.tar.gz && \
    tar zxf libtensorflow-cpu-linux-x86_64-1.0.0.tar.gz -C /usr/local && \
    ldconfig && \
    stack setup && \
    stack test --only-dependencies
