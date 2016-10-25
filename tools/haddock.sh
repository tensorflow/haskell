#!/bin/bash

# Creates shallow haddocks for GitHub pages.

set -eu -o pipefail

IMAGE_NAME=tensorflow/haskell:v0
STACK="stack --docker --docker-image=$IMAGE_NAME"

$STACK haddock --no-haddock-deps tensorflow*
DOC_ROOT=$($STACK path --local-doc-root)
DOCS=docs/haddock
git rm -fr $DOCS
mkdir -p $DOCS
cp $DOC_ROOT/{*.html,*js,*.png,*.gif,*.css} $DOCS
cp -a $DOC_ROOT/tensorflow* $DOCS
rm -f $DOCS/*/*.haddock
git add $DOCS
