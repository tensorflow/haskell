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

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64)
import Data.Functor.Identity (Identity(..))
import Google.Test (googleTest)
import TensorFlow.Types (ListOf(..), Scalar(..))
import TensorFlow.Ops (scalar)
import TensorFlow.Queue
import TensorFlow.Session
    ( asyncProdNodes
    , build
    , buildAnd
    , run
    , runSession
    , run_
    )
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.ByteString as BS

-- | Test basic queue behaviors.
testBasic :: Test
testBasic = testCase "testBasic" $ runSession $ do
    q :: Queue [Int64, BS.ByteString] <- build $ makeQueue 1 ""
    buildAnd run_ $ enqueue q $ 42 :| scalar "Hi" :| Nil
    x <- buildAnd run (dequeue q)
    liftIO $ (Identity (Scalar 42) :| Identity (Scalar "Hi") :| Nil) @=? x

    buildAnd run_ $ enqueue q $ 56 :| scalar "Bar" :| Nil
    y <- buildAnd run (dequeue q)
    let expected = Identity (Scalar 56) :| Identity (Scalar "Bar") :| Nil
    liftIO $ expected @=? y

-- | Test queue pumping.
testPump :: Test
testPump = testCase "testPump" $ runSession $ do
    (deq, pump) <- build $ do
        q :: Queue [Int64, BS.ByteString] <- makeQueue 2 "ThePumpQueue"
        (,) <$> dequeue q
            <*> enqueue q (31 :| scalar "Baz" :| Nil)
    -- This is a realistic use. The pump inputs are pre-bound to some
    -- nodes that produce values when pumped (e.g. read from a
    -- file).
    run_ (pump, pump)

    (x, y) <- run (deq, deq)
    let expected = Identity (Scalar 31) :| Identity (Scalar "Baz") :| Nil
    liftIO $ expected @=? x
    liftIO $ expected @=? y

testAsync :: Test
testAsync = testCase "testAsync" $ runSession $ do
    (deq, pump) <- build $ do
        q :: Queue [Int64, BS.ByteString] <- makeQueue 2 ""
        (,) <$> dequeue q
            <*> enqueue q (10 :| scalar "Async" :| Nil)
    -- Pumps the queue until canceled by runSession exiting.
    asyncProdNodes pump
    -- Picks up a couple values and verifies they are as expected.
    let expected = Identity (Scalar 10) :| Identity (Scalar "Async") :| Nil
    run deq >>= liftIO . (expected @=?)
    run deq >>= liftIO . (expected @=?)

main :: IO ()
main = googleTest [ testBasic
                  , testPump
                  , testAsync
                  ]
