-- Copyright 2020 TensorFlow authors.
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
{-# LANGUAGE OverloadedStrings #-}

module TensorFlow.Convolution
    ( Padding(..)
    , DataFormat(..)
    , conv2D
    , conv2D'
    , conv2DBackpropFilter
    , conv2DBackpropFilter'
    , conv2DBackpropInput
    , conv2DBackpropInput'
    , conv3D
    , conv3D'
    , conv3DBackpropFilter
    , conv3DBackpropFilter'
    , conv3DBackpropFilterV2
    , conv3DBackpropFilterV2'
    , conv3DBackpropInput
    , conv3DBackpropInput'
    , conv3DBackpropInputV2
    , conv3DBackpropInputV2'
    , depthwiseConv2dNative
    , depthwiseConv2dNative'
    , depthwiseConv2dNativeBackpropFilter
    , depthwiseConv2dNativeBackpropFilter'
    , depthwiseConv2dNativeBackpropInput
    , depthwiseConv2dNativeBackpropInput'
    ) where

import Data.Word (Word16)
import Data.Int (Int32,Int64)
import Data.ByteString (ByteString)
import Lens.Family2 ((.~))

import qualified TensorFlow.BuildOp as TF
import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF

-- TODO: Support other convolution parameters such as stride.

-- | Convolution padding.
data Padding = 
        -- | output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        PaddingValid
        -- | output_spatial_shape[i] = ceil(
        --      (input_spatial_shape[i] -
        --          (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i])
      | PaddingSame

paddingToByteString :: Padding -> ByteString
paddingToByteString x = case x of
    PaddingValid -> "VALID"
    PaddingSame  -> "SAME"

-- | Matrix format.
data DataFormat = ChannelLast  -- ^ Channel is the last dimension (e.g. NWC, NHWC, NDHWC)
                | ChannelFirst -- ^ Channel is the first dimension after N (e.g. NCW, NCHW, NCDHW)

-- TODO: Address 1D convolution.

dataFormat2D :: DataFormat -> ByteString
dataFormat2D x = case x of
    ChannelLast  -> "NHWC"
    ChannelFirst -> "NCHW"

dataFormat3D :: DataFormat -> ByteString
dataFormat3D x = case x of
    ChannelLast  -> "NDHWC"
    ChannelFirst -> "NCDHW"

-- | 2D Convolution with default parameters.
conv2D :: TF.OneOf '[Word16, Double, Float] t
       => TF.Tensor v1 t -- ^ input
       -> TF.Tensor v2 t -- ^ filter
       -> TF.Tensor TF.Build t -- ^ output
conv2D = conv2D' id PaddingValid ChannelLast

conv2D' :: TF.OneOf '[Word16, Double, Float] t
        => TF.OpParams
        -> Padding
        -> DataFormat
        -> TF.Tensor v1 t -- ^ input
        -> TF.Tensor v2 t -- ^ filter
        -> TF.Tensor TF.Build t -- ^ output
conv2D' params padding dataformat = TF.conv2D'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)

-- | 2D convolution backpropagation filter with default parameters.
conv2DBackpropFilter :: TF.OneOf '[Word16, Double, Float] t
                     => TF.Tensor v1 t        -- ^ input
                     -> TF.Tensor v2 Int32    -- ^ filter_sizes
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv2DBackpropFilter = conv2DBackpropFilter' id PaddingValid ChannelLast

conv2DBackpropFilter' :: TF.OneOf '[Word16, Double, Float] t
                      => TF.OpParams
                      -> Padding
                      -> DataFormat
                      -> TF.Tensor v1 t        -- ^ input
                      -> TF.Tensor v2 Int32    -- ^ filter_sizes
                      -> TF.Tensor v3 t        -- ^ out_backprop
                      -> TF.Tensor TF.Build t  -- ^ output
conv2DBackpropFilter' params padding dataformat = TF.conv2DBackpropFilter'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)

-- | 2D convolution backpropagation input with default parameters.
conv2DBackpropInput :: TF.OneOf '[Word16, Double, Float] t
                    => TF.Tensor v1 Int32    -- ^ input_sizes
                    -> TF.Tensor v2 t        -- ^ filter
                    -> TF.Tensor v3 t        -- ^ out_backprop
                    -> TF.Tensor TF.Build t  -- ^ output
conv2DBackpropInput = conv2DBackpropInput' id PaddingValid ChannelLast

conv2DBackpropInput' :: TF.OneOf '[Word16, Double, Float] t
                     => TF.OpParams
                     -> Padding
                     -> DataFormat
                     -> TF.Tensor v1 Int32    -- ^ input_sizes
                     -> TF.Tensor v2 t        -- ^ filter
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv2DBackpropInput' params padding dataformat = TF.conv2DBackpropInput'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)

-- | 3D Convolution with default parameters.
conv3D :: TF.OneOf '[Word16, Double, Float] t
       => TF.Tensor v1 t -- ^ input
       -> TF.Tensor v2 t -- ^ filter
       -> TF.Tensor TF.Build t -- ^ output
conv3D = conv3D' id PaddingValid ChannelLast

conv3D' :: TF.OneOf '[Word16, Double, Float] t
        => TF.OpParams
        -> Padding
        -> DataFormat
        -> TF.Tensor v1 t -- ^ input
        -> TF.Tensor v2 t -- ^ filter
        -> TF.Tensor TF.Build t -- ^ output
conv3D' params padding dataformat = TF.conv3D'
    (params . (TF.opAttr "data_format" .~ dataFormat3D dataformat))
    (paddingToByteString padding)

-- | 3D convolution backpropagation filter with default parameters.
conv3DBackpropFilter :: TF.OneOf '[Word16, Double, Float] t
                     => TF.Tensor v1 t        -- ^ input
                     -> TF.Tensor v2 t        -- ^ filter
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropFilter = conv3DBackpropFilter' id PaddingValid ChannelLast

conv3DBackpropFilter' :: TF.OneOf '[Word16, Double, Float] t
                      => TF.OpParams
                      -> Padding
                      -> DataFormat
                      -> TF.Tensor v1 t        -- ^ input
                      -> TF.Tensor v2 t        -- ^ filter
                      -> TF.Tensor v3 t        -- ^ out_backprop
                      -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropFilter' params padding dataformat = TF.conv3DBackpropFilter'
    (params . (TF.opAttr "data_format" .~ dataFormat3D dataformat))
    (paddingToByteString padding)

-- | 3D convolution backpropagation filter with default parameters.
conv3DBackpropFilterV2 :: TF.OneOf '[Word16, Double, Float] t
                     => TF.Tensor v1 t        -- ^ input
                     -> TF.Tensor v2 Int32    -- ^ filter_sizes
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropFilterV2 = conv3DBackpropFilterV2' id PaddingValid ChannelLast

conv3DBackpropFilterV2' :: TF.OneOf '[Word16, Double, Float] t
                      => TF.OpParams
                      -> Padding
                      -> DataFormat
                      -> TF.Tensor v1 t        -- ^ input
                      -> TF.Tensor v2 Int32    -- ^ filter_sizes
                      -> TF.Tensor v3 t        -- ^ out_backprop
                      -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropFilterV2' params padding dataformat = TF.conv3DBackpropFilterV2'
    (params . (TF.opAttr "data_format" .~ dataFormat3D dataformat))
    (paddingToByteString padding)

-- | 3D convolution backpropagation input with default parameters.
conv3DBackpropInput :: TF.OneOf '[Word16, Double, Float] t
                    => TF.Tensor v1 t        -- ^ input
                    -> TF.Tensor v2 t        -- ^ filter
                    -> TF.Tensor v3 t        -- ^ out_backprop
                    -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropInput = conv3DBackpropInput' id PaddingValid ChannelLast

conv3DBackpropInput' :: TF.OneOf '[Word16, Double, Float] t
                     => TF.OpParams
                     -> Padding
                     -> DataFormat
                     -> TF.Tensor v1 t        -- ^ input
                     -> TF.Tensor v2 t        -- ^ filter
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropInput' params padding dataformat = TF.conv3DBackpropInput'
    (params . (TF.opAttr "data_format" .~ dataFormat3D dataformat))
    (paddingToByteString padding)

-- | 3D convolution backpropagation input with default parameters.
conv3DBackpropInputV2 :: (TF.OneOf '[Word16, Double, Float] t, TF.OneOf '[Int32, Int64] tshape)
                    => TF.Tensor v1 tshape   -- ^ input_sizes
                    -> TF.Tensor v2 t        -- ^ filter
                    -> TF.Tensor v3 t        -- ^ out_backprop
                    -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropInputV2 = conv3DBackpropInputV2' id PaddingValid ChannelLast

conv3DBackpropInputV2' :: (TF.OneOf '[Word16, Double, Float] t, TF.OneOf '[Int32, Int64] tshape)
                     => TF.OpParams
                     -> Padding
                     -> DataFormat
                     -> TF.Tensor v1 tshape   -- ^ input_sizes
                     -> TF.Tensor v2 t        -- ^ filter
                     -> TF.Tensor v3 t        -- ^ out_backprop
                     -> TF.Tensor TF.Build t  -- ^ output
conv3DBackpropInputV2' params padding dataformat = TF.conv3DBackpropInputV2'
    (params . (TF.opAttr "data_format" .~ dataFormat3D dataformat))
    (paddingToByteString padding)

-- | Depth-wise 2D convolution native with default parameters.
depthwiseConv2dNative :: TF.OneOf '[Word16, Double, Float] t
                      => TF.Tensor v1 t -- ^ input
                      -> TF.Tensor v2 t -- ^ filter
                      -> TF.Tensor TF.Build t -- ^ output
depthwiseConv2dNative = depthwiseConv2dNative' id PaddingValid ChannelLast

depthwiseConv2dNative' :: TF.OneOf '[Word16, Double, Float] t
                       => TF.OpParams
                       -> Padding
                       -> DataFormat
                       -> TF.Tensor v1 t -- ^ input
                       -> TF.Tensor v2 t -- ^ filter
                       -> TF.Tensor TF.Build t -- ^ output
depthwiseConv2dNative' params padding dataformat = TF.depthwiseConv2dNative'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)

-- | Depth-wise 2D convolution native backpropagation filter with default parameters.
depthwiseConv2dNativeBackpropFilter :: TF.OneOf '[Word16, Double, Float] t
                                    => TF.Tensor v1 t     -- ^ input
                                    -> TF.Tensor v2 Int32 -- ^ filter_sizes
                                    -> TF.Tensor v3 t     -- ^ out_backprop
                                    -> TF.Tensor TF.Build t  -- ^ output
depthwiseConv2dNativeBackpropFilter = depthwiseConv2dNativeBackpropFilter' id PaddingValid ChannelLast

depthwiseConv2dNativeBackpropFilter' :: TF.OneOf '[Word16, Double, Float] t
                                     => TF.OpParams
                                     -> Padding
                                     -> DataFormat
                                     -> TF.Tensor v1 t        -- ^ input
                                     -> TF.Tensor v2 Int32    -- ^ filter_sizes
                                     -> TF.Tensor v3 t        -- ^ out_backprop
                                     -> TF.Tensor TF.Build t  -- ^ output
depthwiseConv2dNativeBackpropFilter' params padding dataformat = TF.depthwiseConv2dNativeBackpropFilter'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)

-- | Depth-wise 2D convolution native backpropagation input with default parameters.
depthwiseConv2dNativeBackpropInput :: TF.OneOf '[Word16, Double, Float] t
                                   => TF.Tensor v1 Int32 -- ^ input_sizes
                                   -> TF.Tensor v2 t     -- ^ input
                                   -> TF.Tensor v3 t     -- ^ out_backprop
                                   -> TF.Tensor TF.Build t  -- ^ output
depthwiseConv2dNativeBackpropInput = depthwiseConv2dNativeBackpropInput' id PaddingValid ChannelLast


depthwiseConv2dNativeBackpropInput' :: TF.OneOf '[Word16, Double, Float] t
                                    => TF.OpParams
                                    -> Padding
                                    -> DataFormat
                                    -> TF.Tensor v1 Int32 -- ^ input_sizes
                                    -> TF.Tensor v2 t     -- ^ input
                                    -> TF.Tensor v3 t     -- ^ out_backprop
                                    -> TF.Tensor TF.Build t  -- ^ output
depthwiseConv2dNativeBackpropInput' params padding dataformat = TF.depthwiseConv2dNativeBackpropInput'
    (params . (TF.opAttr "data_format" .~ dataFormat2D dataformat))
    (paddingToByteString padding)
