# Image for launching TensorFlow Haskell tests.  Unlike the
# development image in docker/Dockerfile, this one doesn't require
# stack to be installed on the host.  This comes at the expense of
# flexibility.

FROM tensorflow/tensorflow:nightly-devel
MAINTAINER TensorFlow authors <tensorflow-haskell@googlegroups.com>

# The build context directory is the top of the tensorflow-haskell
# tree.
ADD . /tfhs
WORKDIR /tfhs

ARG STACK_RESOLVER

RUN \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 575159689BEFB442 && \
    echo 'deb http://download.fpcomplete.com/ubuntu trusty main'| tee /etc/apt/sources.list.d/fpco.list && \
    apt-get update && \
    apt-get install -y \
        # Avoids /usr/bin/ld: cannot find -ltinfo
        libncurses5-dev \
        # Makes stack viable in the container
        libgmp-dev \
        libcurl3-dev \
        stack && \
    # Installs protoc and the libraries.
    ( cd third_party/tensorflow && \
      tensorflow/tools/ci_build/builds/configured CPU && \
      bazel build -c opt '@protobuf//:protoc' && \
      install -s bazel-bin/external/protobuf/protoc /usr/local/bin && \
      bazel build -c opt '//tensorflow:libtensorflow_c.so' && \
      install bazel-bin/tensorflow/libtensorflow_c.so /usr/local/lib && \
      ldconfig ) && \
    stack setup --resolver=${STACK_RESOLVER} && \
    stack test --resolver=${STACK_RESOLVER} --only-dependencies
