#!/bin/bash

echo "Downloading mnist datasets from web.archive.org"

curl https://web.archive.org/web/20160117040036/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > ../tensorflow-mnist-input-data/data/train-images-idx3-ubyte.gz
curl https://web.archive.org/web/20160117040036/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > ../tensorflow-mnist-input-data/data/train-labels-idx1-ubyte.gz
curl https://web.archive.org/web/20160117040036/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > ../tensorflow-mnist-input-data/data/t10k-images-idx3-ubyte.gz
curl https://web.archive.org/web/20160117040036/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > ../tensorflow-mnist-input-data/data/t10k-labels-idx1-ubyte.gz
