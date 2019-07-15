# Image for launching TensorFlow Haskell tests.  Unlike the
# development image in docker/Dockerfile, this one doesn't require
# stack to be installed on the host.  This comes at the expense of
# flexibility.

FROM tensorflow/tensorflow:1.14.0
LABEL maintainer="TensorFlow authors <tensorflow-haskell@googlegroups.com>"

# The build context directory is the top of the tensorflow-haskell
# tree.
ADD . /tfhs
WORKDIR /tfhs

RUN \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 575159689BEFB442 && \
    apt-get update && \
    apt-get install -y \
        # Required by snappy-frames dependency.
        libsnappy-dev \
        # Avoids /usr/bin/ld: cannot find -ltinfo
        libncurses5-dev \
        # Makes stack viable in the container
        libgmp-dev \
        libcurl3-dev \
        zlib1g-dev \
        # For fetching dependencies
        curl \
        unzip \
        # Required for tcp connections by stack (See: https://github.com/tensorflow/haskell/issues/182)
        netbase \
        && \
    # Installs stack.
    curl -O -L https://github.com/commercialhaskell/stack/releases/download/v1.7.1/stack-1.7.1-linux-x86_64.tar.gz && \
    tar zxf stack-1.7.1-linux-x86_64.tar.gz -C /usr/local/bin stack-1.7.1-linux-x86_64/stack --strip 1 && \
    # Installs protoc and the libraries.
    curl -O -L https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip -d /usr/local protoc-3.2.0-linux-x86_64.zip bin/protoc && \
    chmod 755 /usr/local/bin/protoc && \
    curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz && \
    tar zxf libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz -C /usr/local && \
    ldconfig && \
    stack setup && \
    stack test --only-dependencies
