-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

module TensorFlow.Examples.MNIST.InputData
  ( trainingImageData
  , trainingLabelData
  , testImageData
  , testLabelData
  ) where

import Paths_tensorflow_mnist_input_data (getDataFileName)

-- | Download the files containing the canonical MNIST samples and labels.
trainingImageData, trainingLabelData :: IO FilePath
trainingImageData = getDataFileName "train-images-idx3-ubyte.gz"
trainingLabelData = getDataFileName "train-labels-idx1-ubyte.gz"

testImageData, testLabelData :: IO FilePath
testImageData = getDataFileName "t10k-images-idx3-ubyte.gz"
testLabelData = getDataFileName "t10k-labels-idx1-ubyte.gz"
