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

-- | Testing tracing.
module Main where

import Control.Concurrent.MVar (newEmptyMVar, putMVar, tryReadMVar)
import Data.ByteString.Builder (toLazyByteString)
import Data.ByteString.Lazy (isPrefixOf)
import Data.Default (def)
import Lens.Family2 ((&), (.~))
import Test.Framework (defaultMain)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit (assertBool, assertFailure)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF

testTracing :: IO ()
testTracing = do
    -- Verifies that tracing happens as a side-effect of graph extension.
    loggedValue <- newEmptyMVar
    TF.runSessionWithOptions
        (def & TF.sessionTracer .~ putMVar loggedValue)
        (TF.run_ (TF.scalar (0 :: Float)))
    tryReadMVar loggedValue >>=
        maybe (assertFailure "Logging never happened") expectedFormat
  where expectedFormat x =
            let got = toLazyByteString x in
            assertBool ("Unexpected log entry " ++ show got)
                       ("Session.extend" `isPrefixOf` got)

main :: IO ()
main = defaultMain
    [ testCase "Tracing" testTracing
    ]
