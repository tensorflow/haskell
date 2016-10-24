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

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
-- | Rendering of TensorFlow operations as Haskell functions.

module TensorFlow.OpGen
  ( OpGenFlags(..)
  , docOpList
  , flagParser)
  where

import Prelude hiding (head, tail)

import Control.Monad (guard)
import Data.Char (toLower, toUpper)
import Data.Foldable (toList)
import Data.Maybe (fromMaybe, maybeToList)
import Data.ProtoLens (def, showMessage)
import Data.List.NonEmpty (NonEmpty((:|)), head)
import qualified Data.List.NonEmpty as NE
import Lens.Family2 ((^.), (.~), (&), view)
import Options.Applicative (Parser, help, long, strOption, value)
import Proto.Tensorflow.Core.Framework.OpDef
  ( OpList
  , OpDef
  , OpDef'ArgDef
  , attr
  , description
  , inputArg
  , name
  , numberAttr
  , op
  , outputArg
  , summary
  , type'
  , typeAttr
  )
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import System.FilePath (takeBaseName)
import TensorFlow.OpGen.AttrVal
  (AttrDef
  , AttrCase(..)
  , AttrTemplate(..)
  , Template
  , attrDef
  , attrOriginal
  , attrTemplate
  , templateDefault
  , templateRestrictions
  )
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
  , int
  , parens
  , sep
  , stack
  , strictText
  , tuple
  )
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Semigroup as Semigroup
import Data.Text (Text)

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
        , "{-# LANGUAGE FlexibleInstances #-}"
        , "{-# LANGUAGE OverloadedStrings #-}"
        , "{-# LANGUAGE RankNTypes #-}"
        , "{-# LANGUAGE ScopedTypeVariables #-}"
        , "module" <+> strictText moduleName <+> "where"
        , empty
        , imports
        , empty
        , folddoc (\x y -> x </> empty </> y)
                  (map renderDef $
                   filter (not . flip elem exclusions . view name) $
                   toList $ opList ^. op)
        ]
  where moduleName =
            Text.pack (prefix flags) <> "." <> camelCase
             -- Discards the optional trailing _op_lib
            (fromMaybe shortName (Text.stripSuffix "_op_lib" shortName))
        shortName = Text.pack (takeBaseName $ outputFile flags)
        exclusions = Text.splitOn "," $ Text.pack $ excludeList flags

camelCase s = Text.concat $ map upCase
                          $ filter (/= "ops")
                          $ Text.splitOn "_" s

-- | Upper-case the given text.
upCase :: Text -> Text
upCase = forceCase toUpper

-- | Lower-case the given name, and prevent it from overlapping with a reserved
-- Haskell name.
lowCase :: Text -> Text
lowCase = replaceReservedName . forceCase toLower

forceCase :: (Char -> Char) -> Text -> Text
forceCase convert s = maybe "" (\(c, cs) -> Text.cons (convert c) cs)
                      (Text.uncons s)

imports = stack [
      "import Data.ByteString (ByteString)"
    , "import Data.Complex (Complex)"
    , "import Data.Int (Int8, Int16, Int32, Int64)"
    , "import Data.Word (Word8, Word16)"
    , "import Lens.Family2 ((.~), (&))"
    , "import TensorFlow.Build"
    , "import TensorFlow.BuildOp"
    , "import TensorFlow.Tensor"
    , "import TensorFlow.Types"
      ]

renderDef :: OpDef -> Doc
renderDef d =
  stack [
      haddocks
    , n <+> "::" <+> hang 0 (typeSig d)
    , n <+> hang 0 args <+> "|" <+> funcGuard <+> "=" </>  -- args are indented
            -- the body needs to be indented wrt the name
            indent indentation functionBody
    , extras  -- just for debug
    ]
  where
    n = strictText $ fixOpName (d ^. name)
    args = sep $ [hsName | (_, hsName) <- mandatoryAttrs] ++ tensorArgs
    tensorArgs = [strictText $ lowCase (a ^. name) | a <- d ^. inputArg]
    fixOpName = lowCase
    funcGuard = "eqLengthGuard" <+> brackets (commasep entries)
      where
        entries =
            [ parens $ quotedText nAttr <> comma <+>
              brackets (commasep $ toList $
              NE.map renderTensorName tensorNames)
            | (nAttr, tensorNames) <- Map.toList $ numberAttrMap d
            ]
        renderTensorName x = parens $ quotedText x <> comma <+>
                             "length" <+> strictText x
    -- Uses hang 0 to align the argument vertically on multiple lines.
    functionBody = buildFunction <+> parens (hang 0 (stack buildOpParts))
                                 </> indent indentation (sep tensorArgs)
    buildFunction
        | null outputListsSizes = "buildOp"
        | otherwise = "buildListOp" <+> brackets (commasep outputListsSizes)
    outputListsSizes = [ strictText numberAttrName
                       | o <- d ^. outputArg
                       , let numberAttrName = o ^. numberAttr
                       , not (Text.null numberAttrName) &&
                         numberAttrName `Map.member` mandatoryAttrMap d
                       ]
    buildOpParts =
        "opDef" <+> quotedText (d ^. name) :
        -- Renders tensor arguments.
        [ "& opAttr" <+> quotedText tfName <+>
          ".~ tensorType (undefined ::" <+> strictText hsName <> ")"
        | (tfName, (hsName, _)) <- Map.toList typeMap
        ] ++
        -- Renders mandatory attributes as function parameters.
        [ "& opAttr" <+> dquotes tfName <+> ".~" <+> hsName
        | (tfName, hsName) <- mandatoryAttrs
        ] ++
        -- Renders sizes of tensor list types having number_attr.
        [ "& opAttr" <+> quotedText nAttr <+> ".~" <+>
          "(fromIntegral (length" <+> strictText (head tensorNames) <> ") :: Int64)"
        | (nAttr, tensorNames) <- Map.toList $ numberAttrMap d
        ]
    mandatoryAttrs = [(strictText tf, strictText hs)
                     | (tf, (hs, _, _)) <- Map.toList (mandatoryAttrMap d)
                     ]
    haddocks = "-- |" <+> multilineComment (d ^. summary) (d ^. description)
    extras = enclose "{-\n" "\n-}" $
             strictText $ Text.pack $
             showMessage ((def :: OpDef)
                          & inputArg .~ (d ^. inputArg)
                          & outputArg .~ (d ^. outputArg)
                          & attr .~ (d ^. attr))
    typeMap = opDefTypeMap d

-- | Makes a quoted string doc out of the given text value.
quotedText :: Text.Text -> Doc
quotedText = dquotes . strictText

-- | typeSig renders the type signature of the given OpDef.
typeSig :: OpDef -> Doc
typeSig d =
    foralls <+> constraints <+/>
    signatureFold (mandatoryAttrInputs ++ tensorInputs ++ [outputs])
  where
    foralls | Map.null typeMap = empty
            | otherwise =
              "forall"
              <+> sep (refTypes ++ map (strictText . fst) (Map.elems typeMap))
              <+> "."
    constraints | Map.null typeMap = empty
                | otherwise =
                  tuple (concatMap
                         (\(t, aDef) ->
                           "TensorType" <+> strictText t
                           : maybeToList (oneOfRestrictions aDef t))
                         (Map.elems typeMap)) <+> "=>"
    tensorInputs = zipWith tensorArg refTypes (d ^. inputArg)
    refTypes = map (\x -> "v" <> int x) [1..length (d ^. inputArg)]
    tensorArg refType arg = wrapArg refType arg <+>
                            hang 0 ("-- ^" <+> argComment arg)
    -- Argument type is a list of tensors if number_attr is set;
    -- otherwise it's a single Tensor.
    wrapArg refType arg =
        if Text.null (arg ^. numberAttr) then typ else brackets typ
      where typ = tensorType refType arg
    tensorType refType arg =
      "Tensor" <+> refType <+> maybe directType strictText indirectType
      where
        indirectType = fmap fst (Map.lookup (arg ^. typeAttr) typeMap)
        directType = dtTypeToDoc (arg ^. type')
    outputs =
      case d ^. outputArg of
        []  -> "ControlNode"
        [o] -> wrappedOutput o <+> "-- ^" <+> argComment o
        os  -> renderTupleResult os
    wrappedOutput = wrapArg "Value"
    -- Tuple result case is rendered differently to give
    -- individual elements their own comments.
    renderTupleResult os =
        stack $ [ tuple (map wrappedOutput os)
                , flatten commentSummary
                ] ++ map commentDetails os
      where
        commentSummary = "-- ^" <+> tuple [bold (o ^. name) | o <- os]
        commentDetails o =
          stack [ "--"
                , "-- *" <+> argComment o
                ]
    signatureFold = folddoc (\x y -> x </> "->" <+> y)
    mandatoryAttrInputs = [
      dtTypeToDoc dtType <+>
          hang 0 ("-- ^" <+> argComment' tfName descr)
      | (tfName, (_, dtType, descr)) <- Map.toList $ mandatoryAttrMap d]
    typeMap = opDefTypeMap d

-- | Returns the type restriction for the given tensor type if the
-- set of allowed types is not empty (i.e. restricted).
oneOfRestrictions :: AttrDef -> Text -> Maybe Doc
oneOfRestrictions aDef tName = do
    typs <- onAttrType (^. templateRestrictions) aDef
    guard $ not $ null typs
    let typeList = commasep $ map strictText $
                   Set.toList $ Set.fromList $
                   map dtTypeToHaskell typs
    return $ "OneOf" <+> "'" <> brackets typeList <+> strictText tName

-- | Identifies the attributes used as tensor cardinalities. In such
-- cases a list of tensors is supplied as an input_arg. The number of
-- such inputs is communicated as a separate opAttr.
-- The result key is TensorFlow attribute name and the value is the
-- tensor names which have number_attr set to the result key.
numberAttrMap :: OpDef -> Map.Map Text.Text (NonEmpty Text.Text)
numberAttrMap d = Map.fromListWith (Semigroup.<>) [
    (nAttr, replaceReservedName (inp ^. name) :| [])
    | inp <- d ^. inputArg
    , nAttr <- [inp ^. numberAttr]
    , not (Text.null nAttr)
    ]

argComment :: OpDef'ArgDef -> Doc
argComment arg = argComment' (arg ^. name) (arg ^. description)

argComment' :: Text.Text -> Text.Text -> Doc
argComment' argName argDesc =
    bold argName <> splitMultilineText (":" <+>) argDesc

bold :: Text.Text -> Doc
bold n = strictText ("__" <> n <> "__")

opDefTypeMap :: OpDef -> Map.Map Text.Text (Text.Text, AttrDef)
opDefTypeMap d =
    Map.fromList [(n, (lowCase n, a)) | (n, a) <- attrList d, isType a]

attrList :: OpDef -> [(Text.Text, AttrDef)]
attrList d = [(a ^. name, attrDef a) | a <- d ^. attr]

isType :: AttrDef -> Bool
isType = fromMaybe False . onAttrType (const True)

-- | Applies the given function to the data type. Is this a Prism?
onAttrType :: (Template DataType -> a) -> AttrDef -> Maybe a
onAttrType f x = case x ^. attrTemplate of
    AttrSingle (AttrType a) -> Just (f a)
    _ -> Nothing

-- | mandatoryAttrMap contains the attributes chosen by
-- isMandatoryAttr, excluding those which are derived from list of
-- tensor arguments. The key is the TF name of the attribute. The
-- value tuple is (haskell name, TF type, attribute comment).
mandatoryAttrMap :: OpDef -> Map.Map Text.Text (Text.Text, DataType, Text.Text)
mandatoryAttrMap d =
    Map.fromList [ (n, (lowCase n, dtType, a ^. attrOriginal.description))
                 | (n, a) <- attrList d
                 , Just dtType <- [isMandatoryAttr a]
                 -- Excludes the attributes rendered as list lengths.
                 , n `Map.notMember` numberAttrMap d
                 ]

-- | Inspects the attribute and if it is one of the implemented
-- non-tensor values lacking default, then returns Just the TF type.
isMandatoryAttr :: AttrDef -> Maybe DataType
isMandatoryAttr x =
   case x ^. attrTemplate of
     AttrSingle (AttrBool y)  -> noDefault DT_BOOL y
     AttrSingle (AttrInt64 y) -> noDefault DT_INT64 y
     AttrSingle (AttrFloat y) -> noDefault DT_FLOAT y
     _ -> Nothing
   where
     noDefault typ y = maybe (Just typ) (const Nothing) (y ^. templateDefault)

dtTypeToDoc :: DataType -> Doc
dtTypeToDoc = strictText . dtTypeToHaskell

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
dtTypeToHaskell x =
    Text.pack $ "Unsupported type in dtTypeToHaskell: " ++ show x

-- | haddockComment escapes TensorFlow doc strings into haddock.
-- TODO(gnezdo): deal with the markup.
haddockComment :: Text.Text -> Doc
haddockComment = strictText

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

replaceReservedName :: Text -> Text
replaceReservedName n
    | n `Set.member` reservedKeywords = n <> "'"
    | otherwise = n

indentation = 4

reservedKeywords :: Set.Set Text
reservedKeywords = Set.fromList $
    -- Haskell2010 keywords:
    -- https://www.haskell.org/onlinereport/haskell2010/haskellch2.html#x7-180002.4
    -- We don't include keywords that are allowed to be variable names,
    -- in particular: "as", "forall", and "hiding".
    [ "case"
    , "class"
    , "data"
    , "default"
    , "deriving"
    , "do"
    , "else"
    , "foreign"
    , "if"
    , "import"
    , "in"
    , "infix"
    , "infixl"
    , "infixr"
    , "instance"
    , "let"
    , "module"
    , "newtype"
    , "of"
    , "then"
    , "type"
    , "where"
    ]
    ++  -- Nonstandard extensions
    [ "mdo"   -- RecursiveDo
    , "rec"   -- Arrows, RecursiveDo
    , "proc"  -- Arrows
    ]
