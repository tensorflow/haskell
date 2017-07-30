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
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}

module TensorFlow.Gradient
    ( GradientCompatible
    , gradients
    ) where

import Control.Monad (forM, zipWithM)
import Control.Monad.State.Strict (State, evalState, gets, modify)
import Data.ByteString (ByteString)
import Data.Complex (Complex)
import Data.Default (def)
import Data.Int (Int32, Int64)
import Data.Foldable (foldlM)
import Data.List (foldl', sortBy)
import Data.Map.Strict (Map)
import Data.Maybe (fromMaybe, maybeToList, mapMaybe)
import Data.Ord (comparing)
import Data.ProtoLens.TextFormat (showMessage)
import Data.Set (Set)
import Data.Text (Text)
import Data.Tuple (swap)
import Lens.Family2 (Lens', view, (&), (^.), (.~), (%~))
import Lens.Family2.State.Strict (uses)
import Lens.Family2.Stock (at, intAt)
import Lens.Family2.Unchecked (lens, iso)
import Prelude hiding (sum)
import Text.Printf (printf)
import qualified Data.Graph.Inductive.Basic as FGL
import qualified Data.Graph.Inductive.Graph as FGL
import qualified Data.Graph.Inductive.PatriciaTree as FGL
import qualified Data.Graph.Inductive.Query.DFS as FGL
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as Text

import qualified TensorFlow.GenOps.Core as CoreOps
import TensorFlow.Build
    ( MonadBuild
    , Build
    , build
    , renderedNodeDefs
    , opDef
    , opAttr
    , opInputs
    )
import TensorFlow.BuildOp
import TensorFlow.Ops
    ( addN
    , broadcastGradientArgs
    , expandDims
    , fill
    , matMul
    , matMul'
    , reducedShape
    , reluGrad
    , reshape
    , scalar
    , shape
    , softmaxCrossEntropyWithLogits
    , sum
    , scalarize
    , vector
    , zerosLike
    )
import TensorFlow.Output
    ( NodeName(..)
    , Output(..)
    , OutputIx(..)
    , outputIndex
    )
import TensorFlow.Tensor
    ( Tensor(..)
    , Value
    , render
    , expr
    , Rendered
    , tensorNodeName
    , renderedOutput
    , renderValue
    , ToTensor(..)
    )
import TensorFlow.Types (Attribute, OneOf, TensorType, attrLens)
import Proto.Tensorflow.Core.Framework.NodeDef
    (NodeDef, attr, input, op, name)

type GradientCompatible a =
    -- TODO(fmayle): MaxPoolGrad doesn't support Double for some reason.
    (Num a, OneOf '[ Float, Complex Float, Complex Double ] a)

-- TODO(fmayle): Support control flow.
-- TODO(fmayle): Support gate_gradients-like option to avoid race conditions.
-- TODO(fmayle): Do we need to consider control inputs? See _PendingCount in
-- tensorflow/python/ops/gradients.py.
-- TODO(fmayle): Maybe store the gradient functions and numOutputs on the OpDef.


-- | Gradient of @y@ w.r.t. each element of @xs@.
gradients :: forall a v1 t m . ( MonadBuild m
                               , Rendered t
                               , ToTensor t
                               , GradientCompatible a
                               )
          => Tensor v1 a  -- ^ The output of the graph.
          -> [t a]        -- ^ Tensors for which gradients are computed.
          -> m [Tensor Value a]
gradients y xs = build $ do
    -- The gradients are computed using "reverse accumulation", similarly to
    -- what is described here:
    -- https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule.2C_forward_and_reverse_accumulation
    --
    -- The code is summarised as follows:
    --
    -- 1. Create an fgl graph of the relevant nodes (ops) and edges (tensors).
    -- 2. Initialize the gradient of y to 1 (∂y/∂y = 1) and the rest of tensor's
    --    gradients to nothing.
    -- 3. Process the nodes in reverse topological order (i.e. each node comes
    --    after all of its outputs so that the output gradients for a node have
    --    been completely calculated before it is processed):
    --      a. Record the gradient for each of the node's output tensors (∂y/∂w
    --         for each output tensor w).
    --      b. Calculate the gradient of y w.r.t. each of the node's input
    --         tensors using the gradients of the node's output tensors.
    --
    --         Written differently, for each output tensor w and input tensor v:
    --           ∂y/∂w = ...            (calculated in previous steps)
    --           ∂w/∂v = ...            (op specific)
    --           ∂y/∂v = ∂y/∂w * ∂w/∂v  (technically, if tensor v is an input
    --                                   to multiple nodes, then this is only
    --                                   part of ∂y/∂v)
    --
    -- 4. Lookup the recorded gradient for each x in xs.

    y' <- renderValue y
    let yName = tensorNodeName y'
    yOne <- render $ fill (shape y') (scalar 1)
    -- TODO(fmayle): Move this into Build.hs and call it unsafeNodeDefFromName?
    nodeDefLookup :: (NodeName -> NodeDef) <- uses renderedNodeDefs $
        (\f x -> fromMaybe (error $ "no NodeDef found for " ++ show x) (f x))
        . flip Map.lookup
    let (gr, nodeMap) = createGraph yName nodeDefLookup
    -- Set gradient of y to one.
    -- TODO: nicer
    let initPending :: Map.Map FGL.Node (PendingGradients a)
            = Map.empty & (at (nodeMap Map.! yName)
                                . nonEmpty
                                . outputIxAt (outputIndex $ renderedOutput y')
                                . nonEmpty
                                .~ [yOne]
                                )
    -- Calculate the gradients of y w.r.t. each node in the graph.
    gradientMap <- graphGrads gr initPending
    -- Lookup the gradients for each x.
    forM xs $ \x ->
        let Output i xName = renderedOutput x
        in maybe (render $ zerosLike $ toTensor x) return $ do
            n <- nodeMap ^. at xName
            gradientMap ^. at n . nonEmpty . outputIxAt i

outputIxAt :: OutputIx -> Lens' (IntMap.IntMap v) (Maybe v)
outputIxAt = intAt . unOutputIx

-- | Incomplete gradients of a node's outputs.
--
-- The lists represent partial sums. The key is an OutputIx sans newtype.
type PendingGradients a = IntMap.IntMap [Tensor Value a]

-- | Gradients of a node's outputs. The key is an OutputIx sans newtype.
-- TODO: precache the rendering?
type Gradients a = IntMap.IntMap (Tensor Value a)

-- | Graph of TensorFlow operations.
type Graph = FGL.Gr NodeDef EdgeLabel

-- | Data associated with an edge.
--
-- Pair of
--   1. Output index of a tensor from the source node.
--   2. Input index that the tensor connects to on the destination node.
type EdgeLabel = (OutputIx, OutputIx)


-- | State used for calculating gradients.
data GradientsState a = GradientsState
                      { _gradientsPending :: !(Map FGL.Node (PendingGradients a))
                      , _gradientsResult  :: !(Map FGL.Node (Gradients a))
                      }

gradientsPending :: Lens' (GradientsState a) (Map FGL.Node (PendingGradients a))
gradientsPending = lens _gradientsPending (\x y -> x { _gradientsPending = y })

gradientsResult :: Lens' (GradientsState a) (Map FGL.Node (Gradients a))
gradientsResult = lens _gradientsResult (\x y -> x { _gradientsResult = y })


-- TODO(fmayle): Use something like Data.List.Safe.
-- | Safe version of (!!).
safeIndex :: [a] -> Int -> Maybe a
_      `safeIndex` n | n < 0 = Nothing
[]     `safeIndex` _         = Nothing
(x:_)  `safeIndex` 0         = Just x
(_:xs) `safeIndex` n         = xs `safeIndex` (n-1)

-- Copy of http://hackage.haskell.org/package/lens-3.9.0.2/docs/Control-Lens-Iso.html#v%3anon
anon :: a -> (a -> Bool) -> Lens' (Maybe a) a
anon a p = iso (fromMaybe a) go where
  go b | p b       = Nothing
       | otherwise = Just b

non :: Eq a => a -> Lens' (Maybe a) a
non a = anon a (a==)

-- | Lens that defaults Nothing to mempty.
nonEmpty :: (Monoid (t v), Foldable t) => Lens' (Maybe (t v)) (t v)
nonEmpty = anon mempty null

-- TODO: strictness (e.g., foldlM')

-- | Calculate the gradients for every node in a graph.
graphGrads :: forall a. GradientCompatible a
           => Graph
           -> Map FGL.Node (PendingGradients a)
           -- ^ Initial gradients (usually just 1 for the node of interest).
           -> Build (Map FGL.Node (Gradients a))
graphGrads gr initPending = view gradientsResult <$> foldlM go initState nodeOrder
  where
    initState = GradientsState initPending Map.empty
    -- Reverse topological sort.
    -- TODO(fmayle): Filter out nodes that are not successors of any x in xs to
    -- avoid calculating gradients that won't be used.
    nodeOrder = FGL.topsort $ FGL.grev gr
    go :: GradientsState a -> Int -> Build (GradientsState a)
    go state node = do
        -- Aggregate the accumulated gradients for this node.
        outputGrads <-
                sumPendingGradient (state ^. gradientsPending . at node . nonEmpty)
        if null outputGrads
           then pure state
           else do
              let ctx = FGL.context gr node
              inputGrads <- calculateInputGrads ctx outputGrads gr
              -- Calculate the gradients for each of the node's inputs.
              let nextState = state & gradientsResult %~ Map.insert node outputGrads
              pure $ updatePendingGradients ctx inputGrads nextState

-- | Reduce accumulated gradients for each output to one Tensor.
sumPendingGradient :: GradientCompatible a
                   => PendingGradients a -> Build (Gradients a)
sumPendingGradient = sequence . IntMap.mapMaybe f
  where
    f [] = Nothing
    f [x] = Just (pure x)
    f xs = Just (render $ addN xs)


-- | Calculate the gradients of a node's input tensors.
--
-- This is mostly just a wrapper around opGrad.
calculateInputGrads :: forall a. GradientCompatible a
                    => FGL.Context NodeDef EdgeLabel
                    -> Gradients a  -- ^ Output gradients of the node.
                    -> Graph
                    -> Build [Maybe (Tensor Value a)]
calculateInputGrads (inputEdges, _, nodeDef, _) outputGrads gr = do
    fullOutGrads <- fullOutputGrads (numOutputs nodeDef) (nodeDefName nodeDef)
                        outputGrads
    traverse (traverse render) $ opGrad (nodeDef ^. op) nodeDef inputTensors fullOutGrads
  where
    -- Create a tensor from an edge (technically an Output, but it seems less
    -- confusing to refer to it as a tensor here).
    edgeToTensor :: (EdgeLabel, FGL.Node) -> Output
    edgeToTensor ((i, _), n) =
        case FGL.lab gr n of
            Just edgeNodeDef -> Output i (NodeName $ edgeNodeDef ^. name)
            Nothing -> error $ "calculateInputGrads: missing input node for "
                               ++ Text.unpack (nodeDef ^. name)
    -- Input tensors, sorted by input index.
    inputTensors = map edgeToTensor $ sortBy (comparing (snd . fst)) inputEdges

-- | Convert a Map of gradients to a list, with zeros for missing outputs.
fullOutputGrads :: (TensorType a, Num a)
                => OutputIx  -- ^ Number of outputs.
                -> NodeName
                -> Gradients a
                -> Build [Tensor Value a]
fullOutputGrads n o gs =
    mapM (\i -> maybe (render $ zero i) return (gs ^. outputIxAt i)) [0..n-1]
  where
    -- A tensor of zeros with the same shape as the i'th output.
    zero i = zerosLike $ toT (Output i o)


-- | Update the pending gradients of a node's inputs.
updatePendingGradients :: forall a. (TensorType a, Num a)
                       => FGL.Context NodeDef EdgeLabel
                       -> [Maybe (Tensor Value a)]
                       -- ^ Gradient of each input tensor.
                       -> GradientsState a
                       -> GradientsState a
updatePendingGradients (inputEdges, _, nodeDef, _) inputGrads initState =
    foldl' go initState inputEdges
  where
    go :: GradientsState a -> (EdgeLabel, FGL.Node) -> GradientsState a
    go state ((outIndex, OutputIx inIndex), node) =
        case maybeGradient of
            Nothing -> state
            Just g ->
                -- Add to the list of pending gradients for this tensor.
                state & gradientsPending
                      . at node
                      . nonEmpty
                      . outputIxAt outIndex
                      . nonEmpty
                      %~ (g:)
      where
        badSizeErr = error $ printf "updatePendingGradients: bad input index \
                                    \%d for inputGrads of length %d in %s"
                                    inIndex (length inputGrads)
                                    (show (nodeDef ^. name))
        maybeGradient = fromMaybe badSizeErr (safeIndex inputGrads inIndex)


-- | Create a graph that includes a node and its transitive dependencies.
createGraph :: NodeName -> (NodeName -> NodeDef)
            -> (Graph, Map NodeName FGL.Node)
createGraph nodeName nodeDefLookup = (FGL.nmap nodeDefLookup graph, nodeMap)
  where
    -- Parse a tensor name.
    parseTensorName :: Text -> Maybe (NodeName, OutputIx)
    parseTensorName n
        | Text.null n        = error "parseTensorName: empty name"
        | Text.head n == '^' = Nothing  -- Control edge
        | otherwise          =
            let (nm, indexStr) = Text.breakOn ":" n
                index | Text.null indexStr = 0
                      | otherwise = read $ Text.unpack $ Text.tail indexStr
            in Just (NodeName nm, OutputIx index)

    -- Build a map from node name to outward edges.
    --
    -- The state is the set of visited nodes.
    collect :: Maybe (NodeName, OutputIx, OutputIx)
            -> NodeName
            -> State (Set NodeName)
                     (Map NodeName [(NodeName, OutputIx, OutputIx)])
    collect outgoingEdge nm = do
        let nextLookup = Map.singleton nm (maybeToList outgoingEdge)
        seen <- gets (Set.member nm)
        modify (Set.insert nm)
        if seen
            then pure nextLookup
            else do
                let inputs = nodeDefLookup nm ^. input
                    recurse inIndex (parentName, outIndex) =
                        collect (Just (nm, outIndex, inIndex)) parentName
                subEdgeLookups <-
                    zipWithM recurse [0..] $ mapMaybe parseTensorName inputs
                pure $ Map.unionsWith (++) (nextLookup:subEdgeLookups)

    edgeLookup = evalState (collect Nothing nodeName) Set.empty
    -- Associate an ID with each node name.
    nodeMap = Map.fromList $ zip (Map.keys edgeLookup) [0..]
    -- Create the graph.
    graph = FGL.mkGraph (swap <$> Map.toList nodeMap)
                        [ (nodeMap Map.! n, nodeMap Map.! m, (i, j))
                        | (n, edges) <- Map.toList edgeLookup
                        , (m, i, j) <- edges
                        ]

-- | Function to compute the gradient of y w.r.t. each input.
--
-- Let y be an arbitrary tensor
-- and [w_0, ..., w_n] be the output tensors of a node
-- and [v_0, ..., v_n] be the input tensors of the same node.
--
-- Given [∂y/∂w_0, ..., ∂y/∂w_n] and [v_0, ..., v_n], a GradientFunc computes
-- [∂y/∂v_0, ..., ∂y/∂v_n] for a particular op type.
--
-- A Nothing gradient is equivalent to zero (but allows for short circuiting
-- computation when all the gradients for something are Nothing).
type GradientFunc a = NodeDef
                    -> [Output]
                    -- ^ Input tensors.
                    -> [Tensor Value a]
                    -- ^ Gradient of y w.r.t. each output tensor.
                    -> [Maybe (Tensor Build a)]
                    -- ^ Gradient of y w.r.t. each input tensor.


-- TODO(fmayle): Assert the type is correct.
-- | Create a Tensor from an Output.
toT :: Output -> Tensor Build a
toT = Tensor . pure


-- | Wrapper around `TensorFlow.GenOps.Core.slice` that builds vectors from scalars for
-- simple slicing operations.
flatSlice :: forall v1 t . TensorType t
         => Tensor v1 t    -- ^ __input__
         -> Int32          -- ^ __begin__: specifies the offset into the first dimension of
                           -- 'input' to slice from.
         -> Int32          -- ^ __size__: specifies the number of elements of the first dimension
                           -- of 'input' to slice. If size is -1, all remaining elements in the dimension
                           -- are included in the slice (i.e. this is equivalent to setting
                           -- size = input.dim_size(0) - begin).
         -> Tensor Build t -- ^ __output__
flatSlice t begin size = CoreOps.slice t (vector [begin]) (vector [size])

nodeDefName :: NodeDef -> NodeName
nodeDefName = NodeName . view name

-- | Gradient helper for binary component wise operations
-- See https://github.com/tensorflow/tensorflow/blob/e9de087fa7f59c39bbe12ac2c83c5547c83f746c/tensorflow/core/ops/math_grad.cc#L329
gradForBinaryCwise :: ( OneOf '[ Int32, Int64, Float, Double, Complex Float, Complex Double ] t
                      )
                   => (Tensor v1 t, Tensor v1 t)
                   -> (Tensor v1 t, Tensor v1 t)
                   -> [ Maybe (Tensor Build t) ]
gradForBinaryCwise (x, gx) (y, gy) =
    [ Just dx
    , Just dy ]
  where
    dx = reshape (sum gx rx) sx
    dy = reshape (sum gy ry) sy
    sx = shape x
    sy = shape y
    (rx, ry) = broadcastGradientArgs sx sy

-- | The gradient function for an op type.
--
-- These implementations should match their python counterparts in:
-- third_party/tensorflow/python/ops/*_grad.py
opGrad :: forall a . GradientCompatible a => Text -> GradientFunc a

opGrad "Abs" _ [toT -> x] [dz] = [Just $ expr dz * signum x]
opGrad "Neg" _ [_] [dz] = [Just $ negate $ expr dz]
opGrad "Relu" _ [toT -> x] [dz] = [Just $ reluGrad dz x]
opGrad "ReluGrad" _ [_, toT -> x ] [dz] = [Just $ reluGrad dz x, Just $ CoreOps.zerosLike x]

opGrad "Concat" _ _ix [dy]
    -- Concat concatenates input tensors
    --   x1 of shape s1 = [k1, ..., ki_1, ..., kn]
    --   x2 of shape s2 = [k1, ..., ki_2, ..., kn]
    --    .           .     .          .        .
    --    .           .     .          .        .
    --    .           .     .          .        .
    --   xm of shape sm = [k1, ..., ki_m, ..., kn]
    --  along dimension i to an output tensor
    --   y  of shape sy = [k1, ..., k, ..., kn]
    --  where k = sum ki = sum [ki_1,...,ki_m]
    --
    --  The incoming gradient dy from backpropagation is
    --   simply forwarded split across input tensors yielding dx.
    --   Forwarded gradients have shapes s = [s1, ..., sm].
    | m == 1    = Nothing : [Just $ expr dy]
    | otherwise = Nothing : map Just (dx `reshapeZip` s)
  where
    reshapeZip = zipWith reshape
    dx = CoreOps.splitV (fromIntegral m) dy ki _i
    s  :: [Tensor Build Int32]
    s  = map shape x
    x  :: [Tensor Build a]
    x  = map toT $ tail _ix
    -- i: concat dimension. Adjusted modulo n to handle negative indices.
    _i = toT (head _ix) `CoreOps.floorMod` n
    i  = reshape _i $ vector [1 :: Int32]
    -- sizes along concatenated dimension
    ki :: Tensor Build Int32
    ki = CoreOps.concat 0 $ map (\t -> CoreOps.slice t i $ vector [1 :: Int32]) s
    m  = length x
    n  = CoreOps.rank (head x)

opGrad "Square" _ [toT -> x] [dz] =
    -- TODO(fmayle): Handle complex numbers.
    -- TODO(fmayle): The python code makes dz a control dependency of the 2*x
    -- (for performance reasons?). Will need to put these functions in the Build
    -- monad to replicate that.
    [Just $ dz `CoreOps.mul` (2 * x)]

opGrad "Gather" _ [toT -> x, toT -> indices] [dz] =
    -- TODO(fmayle): The python version uses a better performance implementation
    -- when the shape is known without having to run the graph.
    -- TODO(fmayle): We shouldn't convert the result to a dense tensor. Sparse
    -- tensor support will require some thinking.
    [ Just $ CoreOps.unsortedSegmentSum values indices' numRows
    , Nothing
    ]
  where
    -- TODO(gnezdo): Use colocateWith but it requires Build monad.
    denseShape = shape (x :: Tensor Build a)
    numRows = scalarize $ flatSlice denseShape 0 1
    valuesShape = CoreOps.concat 0 [ allDimensions
                                   , flatSlice denseShape 1 (-1)
                                   ]
    values = reshape dz valuesShape
    -- TODO(fmayle): This could be either Int32 or Int64.
    indices' = reshape indices allDimensions :: Tensor Build Int32

opGrad "Max" _ [toT -> x, toT -> indices] [dz] =
    [Just $ indicators `CoreOps.div` numSelected * dz', Nothing]
  where
    sx = shape (x :: Tensor Build a)
    outputShapeKeptDims = reducedShape sx (indices :: Tensor Build Int32)
    y = CoreOps.max x indices
    y' = reshape y outputShapeKeptDims
    dz' = reshape dz outputShapeKeptDims
    indicators = CoreOps.cast $ CoreOps.equal y' x
    numSelected = reshape (sum indicators indices) outputShapeKeptDims

-- Min and Max have identical gradient implementations.
opGrad "Min" u v w = opGrad "Max" u v w

-- Element wise maximum gradient
-- See https://github.com/tensorflow/tensorflow/blob/e9de087fa7f59c39bbe12ac2c83c5547c83f746c/tensorflow/core/ops/math_grad.cc#L473
opGrad "Maximum" _ [toT -> x, toT -> y] [dz] =
    gradForBinaryCwise (x, gx) (y, gy)
  where
    xmask = CoreOps.greaterEqual x y
    gx = CoreOps.select xmask dz (CoreOps.zerosLike dz)
    gy = CoreOps.select (CoreOps.logicalNot xmask) dz (CoreOps.zerosLike dz)

opGrad "Sum" _ [toT -> x, toT -> indices] [dz] =
    [ Just $ CoreOps.tile grad tileScaling, Nothing ]
  where
    -- TODO(gnezdo): Implement the fast-path from math_grad._SumGrad.
    sx = shape (x :: Tensor Build a)
    outputShapeKeptDims = reducedShape sx (indices :: Tensor Build Int32)
    tileScaling = safeShapeDiv sx outputShapeKeptDims
    grad = reshape dz outputShapeKeptDims

opGrad "Mean" u v@[toT -> x, _] w =
    [Just $ dz `CoreOps.div` CoreOps.cast factor, Nothing]
  where
    [Just dz, Nothing] = opGrad "Sum" u v w
    inputShape = shape (x :: Tensor Build a)
    outputShape = shape (dz :: Tensor Build a)
    -- TODO(fmayle): Add fast path when shape is known.
    inputSize = CoreOps.prod inputShape $ rangeOfRank inputShape
    outputSize = CoreOps.prod outputShape $ rangeOfRank outputShape
    factor = safeShapeDiv inputSize outputSize

opGrad "Add" _ [toT -> x, toT -> y] [dz] =
    [ Just $ reshape (sum dz rx) sx
    , Just $ reshape (sum dz ry) sy ]
  where
    sx = shape (x :: Tensor Build a)
    sy = shape (y :: Tensor Build a)
    (rx, ry) = broadcastGradientArgs sx sy

-- Copies the gradients to all inputs
-- Not broadcasting
opGrad "AddN" _ inputs [dz] =
    map ((const . Just . expr) dz) inputs

opGrad "Sub" u v w =
    [Just x, Just (-y)]
  where
    [Just x, Just y] = opGrad "Add" u v w

opGrad "SoftmaxCrossEntropyWithLogits" _ [toT -> x, toT -> y] [dz, _] =
    [ Just $ expandDims dz (-1) * snd (softmaxCrossEntropyWithLogits x y)
    , Nothing ]

opGrad "Mul" _ [toT -> x, toT -> y] [dz] =
    -- TODO(fmayle): Handle complex numbers.
    [ Just $ reshape (sum (dz `CoreOps.mul` y) rx) sx
    , Just $ reshape (sum (x `CoreOps.mul` dz) ry) sy ]
  where
    sx = shape (x :: Tensor Build a)
    sy = shape (y :: Tensor Build a)
    (rx, ry) = broadcastGradientArgs sx sy

opGrad "Div" _ [toT -> x, toT -> y] [dz] =
    -- TODO(fmayle): Handle complex numbers.
    -- TODO(gnezdo): Provide Fractional instance and use '/' instead of div.
    [ Just $ reshape (sum (dz `CoreOps.div` y) rx) sx
    , Just $ reshape (sum (dz `CoreOps.mul` (negate x `CoreOps.div` (y * y)))
                         ry)
                sy
    ]
  where
    sx = shape (x :: Tensor Build a)
    sy = shape (y :: Tensor Build a)
    (rx, ry) = broadcastGradientArgs sx sy

opGrad "MatMul" nodeDef [toT -> x, toT -> y] [dz] =
    let transposeA = lookupAttr nodeDef "transpose_a"
        transposeB = lookupAttr nodeDef "transpose_b"
        transAttrs a b =
            (opAttr "transpose_a" .~ a) . (opAttr "transpose_b" .~ b)
    in case (transposeA, transposeB) of
       (False, False) ->
           [ Just $ matMul' (transAttrs False True) dz y
           , Just $ matMul' (transAttrs True False) x dz]
       (False, True) ->
           [ Just $ matMul dz y
           , Just $ matMul' (transAttrs True False) dz x]
       (True, False) ->
           [ Just $ matMul' (transAttrs False True) y dz
           , Just $ matMul x dz]
       (True, True) ->
           [ Just $ matMul' (transAttrs True True) y dz
           , Just $ matMul' (transAttrs True True) dz x]

opGrad "Transpose" _ [_, toT -> p] [dz] =
    [ Just $ CoreOps.transpose dz
            (CoreOps.invertPermutation p :: Tensor Build Int32)
    , Nothing
    ]

opGrad "Conv2D" nodeDef [toT -> x, toT -> y] [dz] =
    [ Just $ CoreOps.conv2DBackpropInput'
                ((opAttr "strides" .~ strides)
                    . (opAttr "padding" .~ padding)
                    . (opAttr "use_cudnn_on_gpu" .~ useCudnnOnGpu)
                    . (opAttr "data_format" .~ dataFormat))
                (shape x) y dz
    , Just $ CoreOps.conv2DBackpropFilter'
                ((opAttr "strides" .~ strides)
                    . (opAttr "padding" .~ padding)
                    . (opAttr "use_cudnn_on_gpu" .~ useCudnnOnGpu)
                    . (opAttr "data_format" .~ dataFormat))
                x (shape y) dz
    ]
  where
    strides = lookupAttr nodeDef "strides" :: [Int64]
    padding = lookupAttr nodeDef "padding" :: ByteString
    useCudnnOnGpu = lookupAttr nodeDef "use_cudnn_on_gpu" :: Bool
    dataFormat = lookupAttr nodeDef "data_format" :: ByteString

opGrad "MaxPool" nodeDef [toT -> x] [dz] =
    [ Just $ CoreOps.maxPoolGrad'
                ((opAttr "ksize" .~ ksize)
                    . (opAttr "strides" .~ strides)
                    . (opAttr "padding" .~ padding)
                    . (opAttr "data_format" .~ dataFormat))
                x output dz
    ]
  where
    output :: Tensor Build a
    output = toT $ Output 0 (nodeDefName nodeDef)
    ksize = lookupAttr nodeDef "ksize" :: [Int64]
    strides = lookupAttr nodeDef "strides" :: [Int64]
    padding = lookupAttr nodeDef "padding" :: ByteString
    dataFormat = lookupAttr nodeDef "data_format" :: ByteString

opGrad "Reshape" _ [toT -> x, _] [dz] =
    [Just $ reshape dz $ shape (x :: Tensor Build a), Nothing]

opGrad "OneHot" _ _ _ = [Nothing, Nothing, Nothing, Nothing]
opGrad "TruncatedNormal" _ _ _ = [Nothing]

opGrad "RefIdentity" _ _ [dz] = [Just $ expr dz]
opGrad "Cast" nodeDef _ [dz] = [Just reverseCast]
  where
    -- TODO(gnezdo): too permissive, python only allows float types as src_type.
    reverseCast =
        pureOp [] $ pure (opDef "Cast"
                 & opAttr "DstT" .~ (lookupAttr nodeDef "SrcT" :: ByteString)
                 & opAttr "SrcT" .~ (lookupAttr nodeDef "DstT" :: ByteString)
                 & opInputs .~ [renderedOutput dz])

opGrad "DynamicStitch" nodeDef inputs [dz] =
    replicate halfLen Nothing ++ valuesGrads
  where
    halfLen =
        let len = length inputs
            half = len `div` 2
        in if 2 * half == len
           then half
           else error ("Uneven input size " ++ show (len, showMessage nodeDef))
    valuesGrads = [ Just $ CoreOps.gather dz (toT idx :: Tensor Build Int32)
                  | idx <- take halfLen inputs
                  ]

opGrad "DynamicPartition" nodeDef [toT -> xs, toT -> indices] dz =
    [ Just reconstructed, Nothing ]
  where
    reconstructed = CoreOps.reshape stitched
                    (CoreOps.shape (xs :: Tensor Build a) :: Tensor Build Int32)
    stitched = CoreOps.dynamicStitch partitionedIndices dz
    partitionedIndices = CoreOps.dynamicPartition np originalIndices indices
    np = lookupAttr nodeDef "num_partitions" :: Int64
    originalIndices =
        CoreOps.reshape (CoreOps.range 0 (CoreOps.size indices) 1) prefixShape
    prefixShape = shapeInt32 indices
    shapeInt32 t = CoreOps.shape t :: Tensor Build Int32

opGrad "Select" _ [toT -> c, toT -> x, _] [dz] =
    [ Nothing
    , Just $ CoreOps.select c dz zeros
    , Just $ CoreOps.select c zeros dz
    ]
  where zeros = CoreOps.zerosLike x

-- TODO(gnezdo): Unlike Python, no control dependency on dz.
opGrad "Log" _ [toT -> x] [dz] = [ Just $ dz `CoreOps.mul` CoreOps.inv x ]
-- TODO(gnezdo): Reuse the output instead of doing another exp,
-- though, it is probably CSE'd away anyway.
opGrad "Exp" _ [toT -> x] [dz] = [ Just $ dz `CoreOps.mul` CoreOps.exp x ]
opGrad "SparseSegmentSum" _ [toT -> x, toT -> y, toT -> t] [dz] =
    [ Just $ CoreOps.unsortedSegmentSum
             (CoreOps.gather dz (t :: Tensor Build Int32))
             (y :: Tensor Build Int32) inputRows
    , Nothing
    , Nothing
    ]
  where inputRows = flatSlice (shape (x :: Tensor Build a)) 0 1

opGrad "LabelClasses" _ _ _ = [Nothing, Nothing]
opGrad "LabelWeights" _ _ _ = [Nothing]
opGrad "Size" _ _ _ = [Nothing]

-- TODO (jcberentsen): Python implementation uses set_shape for
-- static shape inference, which is unsupported.
-- TODO: implement support for static shape inference
opGrad "Tile" _ [toT -> x, toT -> multiples] [dz] =
    [Just inputGrad, Nothing]
  where
    inputGrad = sum reshapedDz axes
    inputShape = shape (x :: Tensor Build a)
    packed = CoreOps.pack [multiples, inputShape]
    perm = vector [1, 0 :: Int32]
    splitShape = CoreOps.reshape (CoreOps.transpose packed perm) allDimensions
    axes = CoreOps.range 0 (CoreOps.size splitShape) (2 :: Tensor Build Int32)
    reshapedDz = CoreOps.reshape dz splitShape

opGrad "ZerosLike" _ _ _ = [Nothing]
opGrad "Fill" _ _ [dz] = [Nothing, Just $ sum dz rx]
  where
    rx = rangeOfRank dz

-- Treat read ops as an identity function on the variable. This allows us to
-- take gradients w.r.t. to the variable handle instead of the result of a read
-- op. If a variable is read multiple times, the gradients will propagate back
-- through each read.
opGrad "ReadVariableOp" _ _ [dz] = [Just $ expr dz]

-- TODO(fmayle): These can go away if we properly prune the graph.
opGrad "Const" _ _ _ = [Nothing, Nothing]
opGrad "Placeholder" _ _ _ = []
opGrad "VarHandleOp" _ _ _ = []
opGrad "Variable" _ _ _ = []

opGrad n nodeDef ins grads =
    error $ "no gradient implemented for " ++
            show (n, length ins, length grads, showMessage nodeDef, ins)

-- | The number of outputs for an op type.
numOutputs :: NodeDef -> OutputIx
numOutputs o =
    case o ^. op of
        "Abs" -> 1
        "Add" -> 1
        "AddN" -> 1
        "Cast" -> 1
        "Const" -> 1
        "Concat" -> 1
        "Conv2D" -> 1
        "Div" -> 1
        "DynamicStitch" -> 1
        "DynamicPartition" ->
            fromIntegral (lookupAttr o "num_partitions" :: Int64)
        "Exp" -> 1
        "Gather" -> 1
        "LabelClasses" -> 1
        "LabelWeights" -> 1
        "Log" -> 1
        "MatMul" -> 1
        "Max" -> 1
        "Maximum" -> 1
        "MaxPool" -> 1
        "Mean" -> 1
        "Min" -> 1
        "Mul" -> 1
        "Neg" -> 1
        "Placeholder" -> 1
        "OneHot" -> 1
        "ReadVariableOp" -> 1
        "RefIdentity" -> 1
        "Relu" -> 1
        "ReluGrad" -> 1
        "Reshape" -> 1
        "Select" -> 1
        "Size" -> 1
        "SoftmaxCrossEntropyWithLogits" -> 2
        "Square" -> 1
        "SparseSegmentSum" -> 1
        "Sub" -> 1
        "Sum" -> 1
        "Tile" -> 1
        "Transpose" -> 1
        "TruncatedNormal" -> 1
        "VarHandleOp" -> 1
        "Variable" -> 1
        "ZerosLike" -> 1
        "Fill" -> 1
        _ -> error $ "numOuputs not implemented for " ++ show (o ^. op)

-- Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`
safeShapeDiv :: Tensor v1 Int32 -> Tensor v2 Int32 -> Tensor Build Int32
safeShapeDiv x y = x `CoreOps.div` (CoreOps.maximum y 1)

allDimensions :: Tensor Build Int32
allDimensions = vector [-1 :: Int32]

rangeOfRank :: forall v1 t. TensorType t => Tensor v1 t -> Tensor Build Int32
rangeOfRank x = CoreOps.range 0 (CoreOps.rank x) 1

lookupAttr ::  Attribute a1 => NodeDef -> Text -> a1
lookupAttr nodeDef attrName = nodeDef ^. attr . at attrName . non def . attrLens
