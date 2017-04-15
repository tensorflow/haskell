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

{-# LANGUAGE CPP #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{- | Rendering of TensorFlow operations as Haskell functions.

The basic type signature generated for each op is:

> {constraints} => {mandatory attrs} -> {input tensors} -> {output tensors}

where:

* @{mandatory attrs}@ is of the form @A_1 -> ... -> A_N@, where each @A@ is an
 op attribute that doesn't have a default and can't be inferred from other
 inputs.

* @{constraints}@ restrict the type parameters of the input and output tensors
 (for example: 'TensorType' or 'OneOf').

* @{input tensors}@ is of the form @T_1 -> ... -> T_N@, where each @T@ is of
the form @Tensor Ref a@ or @Tensor v a@ (or a list of one of those types),
and @a@ is either a concrete type or a (constrained) type variable.

* @{output tensors}@ is of the form @(T_1,...,T_N)@ for "pure" ops, and
@Build (T_1,...,T_N)@ for "stateful" ops.  An op is considered "stateful" if
it takes a @Tensor Ref@ or @Tensor v ResourceHandle@ as input, or if it's
explicitly marked \"Stateful\" in its @REGISTER_OP@ definition.  (If there
are no outputs, it is either @ControlNode@ or @Build ControlNode@.)
-}

module TensorFlow.OpGen
  ( OpGenFlags(..)
  , docOpList
  , flagParser)
  where

import Data.Foldable (toList)
import Data.Maybe (fromMaybe)
import Data.ProtoLens (def, showMessage)
import Data.List (sortOn)
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NE
import Lens.Family2 ((^.), (.~), (&), view)
import Options.Applicative (Parser, help, long, strOption, value)
import Proto.Tensorflow.Core.Framework.OpDef
  ( OpList
  , OpDef
  , attr
  , inputArg
  , name
  , op
  , outputArg
  )
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import System.FilePath (takeBaseName)
import TensorFlow.OpGen.ParsedOp
import Text.PrettyPrint.Mainland
  ( Doc
  , (<>)
  , (<+>)
  , (</>)
  , (<+/>)
  , brackets
  , comma
  , commasep
  , dquotes
  , empty
  , enclose
  , flatten
  , folddoc
  , hang
  , indent
  , parens
  , sep
  , stack
  , strictText
  , tuple
  )
import qualified Data.Set as Set
import qualified Data.Text as Text

data OpGenFlags = OpGenFlags
     { outputFile :: String
     , prefix :: String
     , excludeList :: String
     }

flagParser :: Parser OpGenFlags
flagParser = OpGenFlags
     <$> strOption (mconcat [ long "output"
                            , help "File to write."
                            ])
     <*> strOption (mconcat [ long "prefix"
                            , help "Haskell package prefix to use"
                            ])
     <*> strOption (mconcat [ long "exclude_list"
                            , value ""
                            , help "Comma separated Ops names to ignore"
                            ])


docOpList :: OpGenFlags -> OpList -> Doc
docOpList flags opList =
  stack [ "{-# LANGUAGE ConstraintKinds #-}"
        , "{-# LANGUAGE DataKinds #-}"
        , "{-# LANGUAGE FlexibleContexts #-}"
        , "{-# LANGUAGE FlexibleInstances #-}"
        , "{-# LANGUAGE OverloadedStrings #-}"
        , "{-# LANGUAGE ScopedTypeVariables #-}"
          -- Avoids reports about shadowing standard library names.
        , "{-# OPTIONS_GHC -fno-warn-name-shadowing #-}"
          -- eqLengthGuard never returns false and dies instead.
        , "{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}"
        , "module" <+> strictText moduleName <+> "where"
        , empty
        , imports
        , empty
        , folddoc (\x y -> x </> empty </> y)
                  (map renderOpAndExtras $
                   sortOn (view name) $
                   filter (not . flip elem exclusions . view name) $
                   toList $ opList ^. op)
        ]
  where moduleName =
            Text.pack (prefix flags) <> "." <> camelCase
             -- Discards the optional trailing _ops_op_lib
            (fromMaybe shortName (Text.stripSuffix "_ops_op_lib" shortName))
        shortName = Text.pack (takeBaseName $ outputFile flags)
        exclusions = Text.splitOn "," $ Text.pack $ excludeList flags
        renderOpAndExtras o = renderOp (parseOp o) </> extras o

imports :: Doc
imports = stack [
      "import Data.ByteString (ByteString)"
    , "import Data.Complex (Complex)"
    , "import Data.Int (Int8, Int16, Int32, Int64)"
    , "import Data.Proxy (Proxy(Proxy))"
    , "import Data.Word (Word8, Word16)"
    , "import Lens.Family2 ((.~), (&))"
    , "import TensorFlow.Build"
    , "import TensorFlow.BuildOp"
    , "import TensorFlow.Tensor"
    , "import TensorFlow.Types"
    ]

renderHaskellName, renderTFName, renderQuotedTFName :: Name -> Doc
renderHaskellName = strictText . unHaskellName . haskellName
renderTFName = strictText . unTFName . tfName
renderQuotedTFName = dquotes . renderTFName


-- | Generate the source code for a single op.
-- For example:
--
-- -- | {haddock comment}
-- foo :: {type sig}
-- foo attr1 attr2 input1 input2 | eqLengthGuard [...] = {function body}
renderOp :: ParsedOp -> Doc
renderOp pOp = stack $
    [ haddocks
    -- Prevent unreasonably long compilation times on ghc-7.10, due
    -- to stack calling "-dump-hi" which (unnecessarily) includes the
    -- inlining information, and is large for ops with many arguments.
#if __GLASGOW_HASKELL__ < 800
    , "{-# NOINLINE" <+> n <+> "#-}"
#endif
    , n <+> "::" <+> hang 0 (typeSig empty pOp)
    , n <+> "=" <+> n <> "' id"
    , n' <+> "::" <+> hang 0 (typeSig "OpParams ->" pOp)
    , n' <+> hang 0 args <+> "|" <+> funcGuard listSizeAttrs
                <+> "=" </>  -- args are indented
                    -- the body needs to be indented wrt the name
                    indent indentation (functionBody pOp)
    ] ++ whereClause listSizeAttrs
  where
    n = renderHaskellName $ parsedOpName pOp
    n' = n <> "'"
    listSizeAttrs = inferredListSizeAttrs pOp
    args = sep $ "op'options"
               : (map renderHaskellName
                    $ map attrName (explicitInputAttrs pOp)
                    ++ map parsedArgName (parsedInputs pOp))
    haddocks = "-- |" <+> multilineComment (parsedOpSummary pOp) (parsedOpDescription pOp)

-- | A check that all lists of the given size have the given length.
-- For example:
--   eqLengthGuard [("N", [("input1", length input1), ("input2", length input2)])]
funcGuard :: [Attr (NonEmpty Name)] -> Doc
funcGuard attrs = "eqLengthGuard" <+> brackets (commasep entries)
      where
        entries =
            [ parens $ nAttr <> comma <+>
              brackets (commasep $ toList $
                            map renderTensorName (toList $ attrInfo a))
            | a <- attrs
            , let nAttr = renderQuotedTFName (attrName a)
            ]
        renderTensorName x = parens $ renderQuotedTFName x <> comma <+>
                        "length" <+> renderHaskellName x

-- | Define the implicit list length attributes.
-- For example:
--   where
--     n1 = fromIntegral (length input1) :: Int64
--     n2 = fromIntegral (length input2) :: Int64
whereClause :: [Attr (NonEmpty Name)] -> [Doc]
whereClause [] = []
whereClause as = [indent 2 $ "where" </> indent 2 (stack $ map defineLengthAttr as)]
  where
    defineLengthAttr a = renderHaskellAttrName a <+> "="
                            <+> "fromIntegral (length"
                            <+> renderHaskellName (NE.head $ attrInfo a)
                            <> ") :: Int64"

renderHaskellAttrName :: Attr a -> Doc
renderHaskellAttrName = renderHaskellName . attrName

functionBody :: ParsedOp -> Doc
functionBody pOp
    | parsedOpIsMonadic pOp
        = "build $ do"
            </> indent indentation (bindOpInputsVar
                        </> "buildOp" <+> outputListsSizes <+> opDef)
    | otherwise
        = "pureOp" <+> outputListsSizes <+> "$ do"
            </> indent indentation (bindOpInputsVar </> "return" <+> opDef)
  where
    outputListsSizes = brackets $ commasep
        [ renderHaskellName a
        | ParsedArg { parsedArgCase = ListArg { argLength = a } }
            <- parsedOutputs pOp
        ]
    opInputsVar = "op'inputs"
    bindOpInputsVar = opInputsVar <+> "<- fmap Prelude.concat $ Prelude.sequence"
                            <+> brackets (commasep $ map (\a -> "buildInputs" <+> a) tensorArgs)
    opDef = parens $ hang 0 $ stack $
        "opDef" <+> renderQuotedTFName (parsedOpName pOp) :
        -- Renders type parameter arguments.
        [ "& opAttr" <+> renderQuotedTFName n <+> ".~" <+> inferredTypeExpr a
        | a <- inferredTypeAttrs pOp, let n = attrName a
        ] ++
        -- Renders mandatory attributes as function parameters.
        [ "& opAttr" <+> renderQuotedTFName n <+> ".~" <+> renderHaskellName n
        | a <- explicitInputAttrs pOp, let n = attrName a
        ] ++
        -- Renders sizes of tensor list types having number_attr.
        [ "& opAttr" <+> renderQuotedTFName n <+> ".~" <+> renderHaskellName n
        | a <- inferredListSizeAttrs pOp, let n = attrName a
        ] ++
        ["& op'options & opInputs .~" <+> opInputsVar]
    tensorArgs = renderTensorArg <$> parsedInputs pOp
    renderTensorArg = renderHaskellName . parsedArgName
    inferredTypeExpr a
        | typeParamIsList $ attrInfo a
            = "fromTensorTypes (Proxy :: Proxy" <+> renderHaskellAttrName a
                    <> ")"
        | otherwise = "tensorType (undefined ::" <+> renderHaskellAttrName a
                            <> ")"

-- | Write a comment with the inputs/outputs/attributes in proto format, for
-- debugging.
extras :: OpDef -> Doc
extras d = enclose "{-\n" "\n-}" $
            strictText $ Text.pack $
            showMessage ((def :: OpDef)
                        & inputArg .~ (d ^. inputArg)
                        & outputArg .~ (d ^. outputArg)
                        & attr .~ (d ^. attr))

-- | The type signature for an op.
-- Of the form:
-- forall t1 t2 v1 v2 . (TensorType t1, TensorType t2)
--      => {pre} Float -> Tensor t1 v1 -> Tensor t2 v2
-- where "Float" is an explicit input attribute, "Tensor t1 v1" is an input, and
-- "Tensor t2 v2" is an output.
typeSig :: Doc -> ParsedOp -> Doc
typeSig pre pOp = constraints
            <+/> pre </> signatureFold (map attrInput (explicitInputAttrs pOp)
                                ++ map tensorArgAndComment (parsedInputs pOp)
                                ++ [outputs])
  where
    constraints
        | null classConstraints = empty
        | otherwise = "forall" <+> sep typeParams <+> "." <+> tuple classConstraints <+> "=>"
    typeParams = [strictText v | k <- parsedInputs pOp ++ parsedOutputs pOp,
                  ArgSomeTensor v <- [argKind $ parsedArgCase k]]
                ++ [renderHaskellAttrName n | n <- inferredTypeAttrs pOp]
                ++ if parsedOpIsMonadic pOp then ["m'"] else []
    -- Use m' as the type parameter to avoid clashing with an attribute name.
    monadConstraint
        | parsedOpIsMonadic pOp = ["MonadBuild m'"]
        | otherwise = []
    classConstraints = monadConstraint ++ map tensorArgConstraint
                                                    (inferredTypeAttrs pOp)
    signatureFold = folddoc (\x y -> x </> "->" <+> y)
    attrInput a = renderAttrType (attrInfo a) <+> hang 0 ("-- ^" <+> attrComment a)
    renderAttrType (AttrSingle a) = renderAttrBaseType a
    renderAttrType (AttrList a) = brackets $ renderAttrBaseType a
    renderAttrBaseType = \case
        AttrBytes -> "ByteString"
        AttrInt64 -> "Data.Int.Int64"
        AttrFloat -> "Float"
        AttrBool -> "Bool"
        AttrType -> "DataType"
        AttrShape -> "Shape"
        AttrTensor -> "TensorProto"

    tensorArgAndComment t = tensorArg t <+> hang 0 ("-- ^" <+> argComment t)
    outputs = case parsedOutputs pOp of
        [] -> wrapOutput "ControlNode"
        -- TODO(judahjacobson): To improve indentation: `tensorArgAndComment a`
        [a] -> wrapOutput (tensorArg a) <+> "-- ^" <+> argComment a
        as -> wrapOutput (tuple (map tensorArg as)) <+/> resultComment as
    wrapOutput o
        | parsedOpIsMonadic pOp = "m'" <+> parens o
        | otherwise = o

-- | Render an op input or output.
-- For example: "Tensor Ref Int64", "Tensor v t"
tensorArg :: ParsedArg -> Doc
tensorArg p = case parsedArgCase p of
    SimpleArg { argType = t, argKind = k } -> tensorType t k
    ListArg { argType = t, argKind = k } -> brackets $ tensorType t k
    MixedListArg {argTypeAttr = t, argKind = k}
        -> "TensorList" <+> parens (kind k) <+> renderHaskellName t
  where
    kind k = case k of
                ArgTensorRef -> "Ref"
                ArgTensorValue -> "Value"
                ArgTensorBuild -> "Build"
                ArgSomeTensor v -> strictText v
    tensorType t k = let
        a = case t of
                ArgTypeFixed dt -> strictText $ dtTypeToHaskell dt
                ArgTypeAttr n -> renderHaskellName n
        in "Tensor" <+> kind k <+> a

attrComment :: Attr a -> Doc
attrComment a = argComment' (attrName a) (attrDescription a)

argComment :: ParsedArg -> Doc
argComment a = argComment' (parsedArgName a) (parsedArgDescription a)

argComment' :: Name -> Text.Text -> Doc
argComment' argName argDesc =
    bold (renderTFName argName) <> splitMultilineText (":" <+>) argDesc

bold :: Doc -> Doc
bold n = "__" <> n <> "__"

-- | Comment for the outputs of an op.
-- For example:
--   -- ^ (__output1__, __output2__)
--   --
--   -- * __output1__: description1
--   --
--   -- * __output2__: description2
resultComment :: [ParsedArg] -> Doc
resultComment os = stack $ flatten commentSummary : map commentDetails os
  where
    commentSummary = "-- ^" <+> tuple [bold (renderTFName $ parsedArgName o) | o <- os]
    commentDetails o =
        stack [ "--"
              , "-- *" <+> argComment o
              ]

-- | Constraints for a given type parameter.
-- E.g.: "TensorType t" or "OneOf [Int64, Float] t"
-- or "TensorTypes ts" or "OneOfs [..] ts".
tensorArgConstraint :: Attr TypeParam -> Doc
tensorArgConstraint a = case attrInfo a of
    TypeParam False Nothing -> "TensorType" <+> n
    TypeParam False (Just as) -> "OneOf" <+> typeList as <+> n
    TypeParam True Nothing -> "TensorTypes" <+> n
    TypeParam True (Just as) -> "OneOfs" <+> typeList as <+> n
  where
    n = renderHaskellAttrName a
    -- Produces a type-level list, e.g.: '[Int32,Int64,Float]
    typeList = ("'" <>) . brackets . commasep . map strictText .
                    Set.toList . Set.fromList .
                    map dtTypeToHaskell . toList

-- NOTE: The cases of this function should be kept in sync with
-- TensorFlow.Types.AllTensorTypes.
dtTypeToHaskell :: DataType -> Text.Text
dtTypeToHaskell DT_BOOL = "Bool"
dtTypeToHaskell DT_BFLOAT16 = "Data.Word.Word16"
dtTypeToHaskell DT_COMPLEX128 = "(Data.Complex.Complex Double)"
dtTypeToHaskell DT_COMPLEX64 = "(Data.Complex.Complex Float)"
dtTypeToHaskell DT_DOUBLE = "Double"
dtTypeToHaskell DT_FLOAT = "Float"
dtTypeToHaskell DT_INT16 = "Data.Int.Int16"
dtTypeToHaskell DT_INT32 = "Data.Int.Int32"
dtTypeToHaskell DT_INT64 = "Data.Int.Int64"
dtTypeToHaskell DT_INT8 = "Data.Int.Int8"
dtTypeToHaskell DT_QINT32 = "Data.Int.Int32"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QINT8 = "Data.Word.Word8"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QINT16 = "Data.Int.Int16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QUINT16 = "Data.Word.Word16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QUINT8 = "Data.Word.Word8"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_STRING = "Data.ByteString.ByteString"
dtTypeToHaskell DT_UINT16 = "Data.Word.Word16"
dtTypeToHaskell DT_HALF = "Data.Word.Word16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_UINT8 = "Data.Word.Word8"
dtTypeToHaskell DT_RESOURCE = "ResourceHandle"
dtTypeToHaskell x =
    Text.pack $ "Unsupported type in dtTypeToHaskell: " ++ show x

-- | haddockComment escapes TensorFlow doc strings into haddock.
-- TODO(gnezdo): deal with the markup.
haddockComment :: Text.Text -> Doc
haddockComment = strictText

-- | Generate a multiline comment.  For example:
--   summary'
--   --
--   -- detail_line1
--   -- detail_line2
--   -- ...
multilineComment :: Text.Text -> Text.Text -> Doc
multilineComment summary' detail =
    haddockComment summary' </>
    splitMultilineText insertParagraphAndComment detail
  where insertParagraphAndComment x = "--" </> "--" <+> x

-- | Converts the given multi-line detail string into
-- a multi-line haddock. Applies the given lead to the
-- first line. Returns an empty document for empty detail.
splitMultilineText :: (Doc -> Doc) -> Text.Text -> Doc
splitMultilineText lead detail =
  case Text.lines detail of
    [] -> empty
    (l : ls) -> stack $ lead (haddockComment l)
                      : map (("--" <+>) . haddockComment) ls

indentation :: Int
indentation = 4
