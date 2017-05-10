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

{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- Purposely disabled to confirm doubleFuncNoSig can be written without type.
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

import Control.Monad (replicateM)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64)
import Test.Framework (defaultMain, Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.Framework.Providers.QuickCheck2 (testProperty)
import Test.HUnit ((@=?))
import Test.QuickCheck (Arbitrary(..), listOf, suchThat)
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as B8
import qualified Data.Vector as V

import qualified TensorFlow.GenOps.Core as TF (select)
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF

instance Arbitrary B.ByteString where
    arbitrary = B.pack <$> arbitrary

-- Test encoding tensors, feeding them through tensorflow, and decoding the
-- results.
testFFIRoundTrip :: Test
testFFIRoundTrip = testCase "testFFIRoundTrip" $
    TF.runSession $ do
        let floatData = V.fromList [1..6 :: Float]
            stringData = V.fromList [B8.pack (show x) | x <- [1..6::Integer]]
            boolData = V.fromList [True, True, False, True, False, False]
        f <- TF.placeholder [2,3]
        s <- TF.placeholder [2,3]
        b <- TF.placeholder [2,3]
        let feeds = [ TF.feed f (TF.encodeTensorData [2,3] floatData)
                    , TF.feed s (TF.encodeTensorData [2,3] stringData)
                    , TF.feed b (TF.encodeTensorData [2,3] boolData)
                    ]
        -- Do something idempotent to the tensors to verify that tensorflow can
        -- handle the encoding. Originally this used `TF.identity`, but that
        -- wasn't enough to catch a bug in the encoding of Bool.
        (f', s', b') <- TF.runWithFeeds feeds
                            (f `TF.add` 0, TF.identity s, TF.select b b b)
        liftIO $ do
            floatData @=? f'
            stringData @=? s'
            boolData @=? b'


data TensorDataInputs a = TensorDataInputs [Int64] (V.Vector a)
    deriving Show

instance Arbitrary a => Arbitrary (TensorDataInputs a) where
    arbitrary = do
        -- Limit the size of the final vector, and also guard against overflow
        -- (i.e., p<0) when there are too many dimensions
        let validProduct p = p > 0 && p < 100
        sizes <- listOf (arbitrary `suchThat` (>0))
                    `suchThat` (validProduct . product)
        elems <- replicateM (fromIntegral $ product sizes) arbitrary
        return $ TensorDataInputs sizes (V.fromList elems)

-- Test that a vector is unchanged after being encoded and decoded.
encodeDecodeProp :: (TF.TensorDataType V.Vector a, Eq a) => TensorDataInputs a -> Bool
encodeDecodeProp (TensorDataInputs shape vec) =
    TF.decodeTensorData (TF.encodeTensorData (TF.Shape shape) vec) == vec

testEncodeDecodeQcFloat :: Test
testEncodeDecodeQcFloat = testProperty "testEncodeDecodeQcFloat"
    (encodeDecodeProp :: TensorDataInputs Float -> Bool)

testEncodeDecodeQcInt64 :: Test
testEncodeDecodeQcInt64 = testProperty "testEncodeDecodeQcInt64"
    (encodeDecodeProp :: TensorDataInputs Int64 -> Bool)

testEncodeDecodeQcString :: Test
testEncodeDecodeQcString = testProperty "testEncodeDecodeQcString"
    (encodeDecodeProp :: TensorDataInputs B.ByteString -> Bool)

doubleOrInt64Func :: TF.OneOf '[Double, Int64] a => a -> a
doubleOrInt64Func = id

doubleOrFloatFunc :: TF.OneOf '[Double, Float] a => a -> a
doubleOrFloatFunc = id

doubleFunc :: TF.OneOf '[Double] a => a -> a
doubleFunc = doubleOrFloatFunc . doubleOrInt64Func

-- No explicit type signature; make sure it can be inferred automatically.
-- (Note: this would fail if we didn't have NoMonomorphismRestriction, since it
-- can't simplify the type all the way to `Double -> Double`.
doubleFuncNoSig = doubleOrFloatFunc . doubleOrInt64Func

typeConstraintTests :: Test
typeConstraintTests = testCase "type constraints" $ do
    42 @=? doubleOrInt64Func (42 :: Double)
    42 @=? doubleOrInt64Func (42 :: Int64)
    42 @=? doubleOrFloatFunc (42 :: Double)
    42 @=? doubleOrFloatFunc (42 :: Float)
    42 @=? doubleFunc (42 :: Double)
    42 @=? doubleFuncNoSig (42 :: Double)


main :: IO ()
main = defaultMain
            [ testFFIRoundTrip
            , testEncodeDecodeQcFloat
            , testEncodeDecodeQcInt64
            , testEncodeDecodeQcString
            , typeConstraintTests
            ]
