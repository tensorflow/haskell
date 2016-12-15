-- | This module helps parse the proto OpDef into a Haskell type which is more
-- descriptive of how the attributes and arguments will be used in the
-- generated code.
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
module TensorFlow.OpGen.ParsedOp
    ( ParsedOp(..)
    , Name(..)
    , HaskellName(..)
    , TFName(..)
    , Attr(..)
    , AttrType(..)
    , AttrBaseType(..)
    , ParsedArg(..)
    , ParsedArgCase(..)
    , ArgType(..)
    , ArgKind(..)
    , parseOp
    , camelCase
    ) where

import Data.Char (toUpper, toLower)
import Data.List (sortBy)
import Data.List.NonEmpty (NonEmpty, nonEmpty)
import Data.Maybe (mapMaybe)
import Data.Monoid ((<>))
import Data.Ord (comparing)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as Text
import Lens.Family2 ((^.))
import Proto.Tensorflow.Core.Framework.AttrValue (list)
import Proto.Tensorflow.Core.Framework.OpDef
    ( OpDef
    , OpDef'ArgDef
    , OpDef'AttrDef
    , allowedValues
    , attr
    , maybe'defaultValue
    , description
    , name
    , inputArg
    , isRef
    , isStateful
    , outputArg
    , summary
    , typeListAttr
    , numberAttr
    , typeAttr
    , type'
    )
import Proto.Tensorflow.Core.Framework.Types (DataType(DT_RESOURCE))

data ParsedOp = ParsedOp
    { parsedOpName :: Name
    , parsedOpSummary :: Text
    , parsedOpDescription :: Text
    , parsedInputs :: [ParsedArg]
    , parsedOutputs :: [ParsedArg]
    , explicitInputAttrs :: [Attr AttrType]
        -- ^ Attributes that must be set explicitly when creating the op.
        -- Associated with the type of the attribute.
    , inferredTypeAttrs :: [Attr [DataType]]
        -- ^ Attributes that are type parameters.
        -- Associated with the list of allowed types (see: TensorFlow.Types.OneOf).
        -- If this list is empty, then any type is acceptable.
    , inferredListSizeAttrs :: [Attr (NonEmpty Name)]
        -- Attributes which are list sizes (ints) that are inferred automatically
        -- from one or more of the input tensors.
        -- Associated with the list of tensors whose size it describes.
    , parsedOpIsMonadic :: Bool
        -- ^ Whether this op is stateful or takes a stateful input.  Such ops
        -- should not be CSE'd and must be monadic in our API (i.e., return a
        -- Build action).
    }

data Name = Name
    { haskellName :: HaskellName
    , tfName :: TFName
    }

-- | A raw name as specified in the OpDef proto.
newtype TFName = TFName { unTFName :: Text }
    deriving (Eq, Ord)

-- | A name that's appropriate for a variable in a Haskell source file.
newtype HaskellName = HaskellName { unHaskellName :: Text }

-- | A named attribute, associated with some information about it.
data Attr a = Attr
    { attrName :: Name
    , attrDescription :: Text
    , attrInfo :: a
    }

-- | The type of an attribute.
data AttrType = AttrSingle AttrBaseType
                | AttrList AttrBaseType
                deriving Eq

data AttrBaseType = AttrBytes | AttrInt64 | AttrFloat | AttrBool
                | AttrType | AttrShape | AttrTensor
                deriving Eq

-- | An input or output argument (Tensor) for an op.
data ParsedArg = ParsedArg
    { parsedArgName :: Name
    , parsedArgDescription :: Text
    , parsedArgCase :: ParsedArgCase
    , parsedArgKind :: ArgKind
    }

data ParsedArgCase
    = SimpleArg { argType :: ArgType }
    | ListArg
        { argLength :: Name  -- ^ The attribute that specifies this list's length.
        , argType :: ArgType
        }
    | MixedListArg { argTypeAttr :: Name }
        -- ^ A heterogeneous list.
        -- TODO(judahjacobson): Implement this.

-- | The type of an argument.
data ArgType
    = ArgTypeFixed DataType -- ^ A fixed type.
    | ArgTypeAttr Name  -- ^ A type that depends on an attribute.

-- The kind of an op input or output (not including the argument type `a`).
data ArgKind
    = ArgTensorRef -- Tensor Ref a
    | ArgTensorValue -- Tensor Value a
    | ArgTensorEither Text -- Tensor v a; the Text is the variable `v`
    | ArgResource -- Resource a

isRefKind :: ArgKind -> Bool
isRefKind ArgTensorRef = True
isRefKind ArgResource = True
isRefKind _ = False

makeName :: Text -> Name
makeName n = Name
    { haskellName = HaskellName $ fixReservedName $ lowCase n
    , tfName = TFName n
    }

-- | Change a name so it doesn't conflict with any Haskell keywords.
fixReservedName :: Text -> Text
fixReservedName n
    | n `Set.member` reservedKeywords = n <> "'"
    | otherwise = n

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

-- | Lower-case the given text.
lowCase :: Text -> Text
lowCase = forceCase toLower

forceCase :: (Char -> Char) -> Text -> Text
forceCase convert s = maybe "" (\(c, cs) -> Text.cons (convert c) cs)
                      (Text.uncons s)

camelCase :: Text -> Text
camelCase s = Text.concat $ map upCase
                          $ Text.splitOn "_" s

-- | Upper-case the given text.
upCase :: Text -> Text
upCase = forceCase toUpper


parseOp :: OpDef -> ParsedOp
parseOp o = ParsedOp
    { parsedOpName = makeName $ o ^. name
    , parsedOpSummary = o ^. summary
    , parsedOpDescription = o ^. description
    , parsedOpIsMonadic = o ^. isStateful
                            || any (isRefKind . parsedArgKind) parsedInputs
    , ..
    }
  where
    parsedInputs = zipWith (\a v -> parseArg a (inputTensorKind a v))
                (o ^. inputArg) tensorKindParams
    tensorKindParams = ["v" <> Text.pack (show x) | x <- [1::Integer ..]]
    parsedOutputs = map (\a -> parseArg a (outputTensorKind a)) (o ^. outputArg)
    -- Type attributes that can be inferred from at least one input or output.
    argTypeAttrs = Set.fromList $ mapMaybe parsedArgTypeAttr
                       $ parsedInputs ++ parsedOutputs
    inferredTypeAttrs = filter ((`Set.member` argTypeAttrs) . tfName . attrName)
                            $ mapMaybeAttrs getInferredTypeAttr $ o ^. attr
    -- Integer attributes that can be inferred from the size of at least one
    -- input list.
    inferredListSizeAttrs = mapMaybeAttrs (getInferredListSizeAttr parsedInputs)
                                $ o ^. attr
    implicitAttrs = Set.fromList $ map tfName $
                        map attrName inferredTypeAttrs
                            ++ map attrName inferredListSizeAttrs
    -- Attributes that can't be inferred and don't have defaults, so must be passed
    -- as separate arguments to the op.
    explicitInputAttrs = sortBy (comparing (tfName . attrName))
                        $ mapMaybeAttrs (getExplicitInputAttr implicitAttrs)
                        $ o ^. attr

-- TODO(judahjacobson): Some arguments should be refs.
inputTensorKind :: OpDef'ArgDef -> Text -> ArgKind
inputTensorKind a v
    | a ^. type' == DT_RESOURCE = ArgResource
    | a ^. isRef = ArgTensorRef
    | otherwise = ArgTensorEither v

outputTensorKind :: OpDef'ArgDef -> ArgKind
outputTensorKind a
    | a ^. type' == DT_RESOURCE = ArgResource
    | a ^. isRef = ArgTensorRef
    | otherwise = ArgTensorValue

getExplicitInputAttr :: Set.Set TFName -> OpDef'AttrDef -> Maybe AttrType
getExplicitInputAttr implicitAttrs a
    | TFName (a ^. name) `Set.notMember` implicitAttrs
    , a ^. maybe'defaultValue == Nothing
    , t <- parseAttrType (a ^. type')
    , t `elem` map AttrSingle [AttrBool, AttrInt64, AttrFloat, AttrShape] = Just t
    | otherwise = Nothing

-- | The type attribute used by this input or output (if any).
parsedArgTypeAttr :: ParsedArg -> Maybe TFName
parsedArgTypeAttr p = case parsedArgCase p of
    SimpleArg {argType = t} -> fromArgType t
    ListArg {argType = t} -> fromArgType t
    MixedListArg {argTypeAttr = n} -> Just $ tfName n
  where
    fromArgType (ArgTypeAttr n) = Just $ tfName n
    fromArgType _ = Nothing

getInferredTypeAttr :: OpDef'AttrDef -> Maybe [DataType]
getInferredTypeAttr a
    | a ^. type' == "type" = Just $ a ^. allowedValues . list . type'
    | otherwise = Nothing

getInferredListSizeAttr :: [ParsedArg] -> OpDef'AttrDef -> Maybe (NonEmpty Name)
getInferredListSizeAttr inputs a
    | a ^. type' == "int"
        = nonEmpty [t | ParsedArg { parsedArgName = t
                                  , parsedArgCase
                                        = ListArg { argLength = n }
                                  } <- inputs
                      , TFName (a ^. name) == tfName n]
    | otherwise = Nothing
    
-- | Like mapMaybe, but associates the attribute name/description with the given info.
mapMaybeAttrs :: (OpDef'AttrDef -> Maybe a) -> [OpDef'AttrDef] -> [Attr a]
mapMaybeAttrs f = mapMaybe $ \a -> do
                            x <- f a
                            Just Attr
                                { attrName = makeName (a ^. name)
                                , attrDescription = a ^. description
                                , attrInfo = x
                                }
  
parseArg :: OpDef'ArgDef -> ArgKind -> ParsedArg
parseArg a tKind = ParsedArg
    { parsedArgName = makeName (a ^. name)
    , parsedArgDescription = a ^. description
    , parsedArgCase = parseArgCase a
    , parsedArgKind = tKind
    }

parseArgCase :: OpDef'ArgDef -> ParsedArgCase
parseArgCase a
    | Just n <- maybeAttr (a ^. typeListAttr) = MixedListArg n
    | Just n <- maybeAttr (a ^. numberAttr) = ListArg n thisArgType
    | otherwise = SimpleArg thisArgType
  where
    thisArgType
        | Just n <- maybeAttr (a ^. typeAttr) = ArgTypeAttr n
        | a ^. type' == DT_RESOURCE = ArgTypeAttr (makeName "dtype")
        | otherwise = ArgTypeFixed (a ^. type')
    maybeAttr :: Text -> Maybe Name
    maybeAttr "" = Nothing
    maybeAttr t = Just $ makeName t

parseAttrType :: Text -> AttrType
parseAttrType = \case
    "string" -> AttrSingle AttrBytes
    "int" -> AttrSingle AttrInt64 
    "float" -> AttrSingle AttrFloat 
    "bool" -> AttrSingle AttrBool 
    "type" -> AttrSingle AttrType 
    "shape" -> AttrSingle AttrShape 
    "tensor" -> AttrSingle AttrTensor 
    "list(string)" -> AttrList AttrBytes
    "list(int)" -> AttrList AttrInt64
    "list(float)" -> AttrList AttrFloat
    "list(bool)" -> AttrList AttrBool
    "list(type)" -> AttrList AttrType
    "list(shape)" -> AttrList AttrShape
    "list(tensor)" -> AttrList AttrTensor
    t -> error $ "parseAttrType: unrecognized type " ++ show t
