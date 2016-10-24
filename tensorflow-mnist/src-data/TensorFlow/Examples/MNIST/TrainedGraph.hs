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

{-# LANGUAGE OverloadedStrings #-}
-- | Paths to test helper files.
module TensorFlow.Examples.MNIST.TrainedGraph where

import Paths_tensorflow_mnist (getDataFileName)
import Data.ByteString (ByteString)
import Data.ByteString.Char8 (pack)

-- | File containing a Tensorflow serialized proto of MNIST.
mnistPb :: IO FilePath
mnistPb = getDataFileName "data/MNIST.pb"

-- | Files containing pre-trained weights for MNIST.
wtsCkpt, biasCkpt :: IO ByteString
wtsCkpt = pack <$> getDataFileName "data/MNISTWts.ckpt"
biasCkpt = pack <$> getDataFileName "data/MNISTBias.ckpt"
