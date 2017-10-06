#!/bin/bash

# This wrapper is necessary to fool 'stack --docker' into
# using nvidia-docker which again will invoke the real
# docker in the end.
#
# Symlink this file as 'docker' somewhere in path

NV_DOCKER='/usr/bin/docker' nvidia-docker "$@"
