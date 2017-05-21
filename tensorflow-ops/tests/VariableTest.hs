{-# LANGUAGE OverloadedLists #-}
module Main (main) where

import Data.Maybe (isJust)
import Control.Monad (when)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Vector.Storable as V
import TensorFlow.Core
    ( unScalar
    , render
    , run_
    , runSession
    , run
    , withControlDependencies)
import qualified TensorFlow.Ops as Ops
import TensorFlow.Variable
    ( Variable
    , readValue
    , initializedValue
    , initializedVariable
    , assign
    , assignAdd
    , variable
    )
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?), assertFailure)

main :: IO ()
main = defaultMain
            [ testInitializedVariable
            , testInitializedVariableShape
            , testInitializedValue
            , testDependency
            , testRereadRef
            , testAssignAdd
            ]

testInitializedVariable :: Test
testInitializedVariable =
    testCase "testInitializedVariable" $ runSession $ do
        (formula, reset) <- do
            v <- initializedVariable 42
            r <- assign v 24
            return (1 + readValue v, r)
        result <- run formula
        liftIO $ 43 @=? (unScalar result :: Float)
        run_ reset  -- Updates v to a different value
        rerunResult <- run formula
        liftIO $ 25 @=? (unScalar rerunResult :: Float)

testInitializedVariableShape :: Test
testInitializedVariableShape =
    testCase "testInitializedVariableShape" $ runSession $ do
        vector <- initializedVariable (Ops.constant [1] [42 :: Float])
        result <- run (readValue vector)
        liftIO $ [42] @=? (result :: V.Vector Float)

testInitializedValue :: Test
testInitializedValue =
    testCase "testInitializedValue" $ runSession $ do
        initialized <- initializedVariable (Ops.constant [1] [42 :: Float])
        result <- run (initializedValue initialized)
        liftIO $ Just [42] @=? (result :: Maybe (V.Vector Float))

        uninitialized <- variable [1]
        -- Can't use @=? because there is no Show instance for Tensor.
        when (isJust (initializedValue (uninitialized :: Variable Float))) $
            liftIO $ assertFailure "initializedValue should be Nothing, got Just"

testDependency :: Test
testDependency =
    testCase "testDependency" $ runSession $ do
        v <- variable []
        a <- assign v 24
        r <- withControlDependencies a $ render (readValue v + 18)
        result <- run r
        liftIO $ (42 :: Float) @=? unScalar result

-- | See https://github.com/tensorflow/haskell/issues/92.
-- Even though we're not explicitly evaluating `f0` until the end,
-- it should hold the earlier value of the variable.
testRereadRef :: Test
testRereadRef = testCase "testReRunAssign" $ runSession $ do
    w <- initializedVariable 0
    f0 <- run (readValue w)
    run_ =<< assign w (Ops.scalar (0.1 :: Float))
    f1 <- run (readValue w)
    liftIO $ (0.0, 0.1) @=? (unScalar f0, unScalar f1)

testAssignAdd :: Test
testAssignAdd = testCase "testAssignAdd" $ runSession $ do
    w <- initializedVariable 42
    run_ =<< assignAdd w 17
    f1 <- run (readValue w)
    liftIO $ (42 + 17 :: Float) @=? unScalar f1
