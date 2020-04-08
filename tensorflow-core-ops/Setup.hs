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

-- | Generates the wrappers for Ops shipped with tensorflow.
module Main where

import Distribution.PackageDescription
    ( PackageDescription(..)
    , libBuildInfo
    , hsSourceDirs
    )
import qualified Distribution.Simple.BuildPaths as BuildPaths
import Distribution.Simple.LocalBuildInfo (LocalBuildInfo)
import Distribution.Simple
    ( defaultMainWithHooks
    , simpleUserHooks
    , UserHooks(..)
    )
import Data.List (intercalate)
import Data.ProtoLens (decodeMessage)
import System.Directory (createDirectoryIfMissing)
import System.Exit (exitFailure)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import TensorFlow.Internal.FFI (getAllOpList)
import TensorFlow.OpGen (docOpList, OpGenFlags(..))
import Text.PrettyPrint.Mainland (prettyLazyText)
import qualified Data.Text.Lazy.IO as Text

main = defaultMainWithHooks generatingOpsWrappers

-- TODO: Generalize for user libraries by replacing getAllOpList with
-- a wrapper around TF_LoadLibrary. The complicated part is interplay
-- between bazel and Haskell build system.
generatingOpsWrappers :: UserHooks
generatingOpsWrappers = hooks
    { buildHook = \p l h f -> generateSources l >> buildHook hooks p l h f
    , haddockHook = \p l h f -> generateSources l >> haddockHook hooks p l h f
    , replHook = \p l h f args -> generateSources l
                                        >> replHook hooks p l h f args
    }
  where
    flagsBuilder dir = OpGenFlags
        { outputFile = dir </> "Core.hs"
        , prefix = "TensorFlow.GenOps"
        , excludeList = intercalate "," blackList
        }
    hooks = simpleUserHooks
    generateSources :: LocalBuildInfo -> IO ()
    generateSources l = do
        let dir = autogenModulesDir l </> "TensorFlow/GenOps"
        createDirectoryIfMissing True dir
        let flags = flagsBuilder dir
        pb <- getAllOpList
        case decodeMessage pb of
            Left e -> hPutStrLn stderr e >> exitFailure
            Right x -> Text.writeFile (outputFile flags)
                                      (prettyLazyText 80 $ docOpList flags x)

-- | Add the autogen directory to the hs-source-dirs of all the targets in the
-- .cabal file.  Used to fool 'sdist' by pointing it to the generated source
-- files.
fudgePackageDesc
    :: LocalBuildInfo -> PackageDescription -> PackageDescription
fudgePackageDesc lbi p = p
    { library =
        (\lib -> lib { libBuildInfo = fudgeBuildInfo (libBuildInfo lib) })
            <$> library p
    }
  where
    fudgeBuildInfo bi =
        bi { hsSourceDirs = autogenModulesDir lbi : hsSourceDirs bi }

blackList =
    [ -- Requires the "func" type:
      "FilterDataset"
    , "BatchFunction"
    , "Case"
    , "ChooseFastestBranchDataset"
    , "ExperimentalGroupByReducerDataset"
    , "ExperimentalGroupByWindowDataset"
    , "ExperimentalMapAndBatchDataset"
    , "ExperimentalMapDataset"
    , "ExperimentalNumaMapAndBatchDataset"
    , "ExperimentalParallelInterleaveDataset"
    , "ExperimentalScanDataset"
    , "ExperimentalTakeWhileDataset"
    , "FilterDataset"
    , "FlatMapDataset"
    , "For"
    , "GeneratorDataset"
    , "GroupByReducerDataset"
    , "GroupByWindowDataset"
    , "If"
    , "InterleaveDataset"
    , "MapAndBatchDataset"
    , "MapAndBatchDatasetV2"
    , "MapDataset"
    , "MapDefun"
    , "OneShotIterator"
    , "ParallelInterleaveDataset"
    , "ParallelInterleaveDatasetV2"
    , "ParallelMapDataset"
    , "ParseSequenceExample"
    , "PartitionedCall"
    , "ReduceDataset"
    , "RemoteCall"
    , "ScanDataset"
    , "StatefulPartitionedCall"
    , "StatelessIf"
    , "StatelessWhile"
    , "SymbolicGradient"
    , "TPUPartitionedCall"
    , "TPUReplicate"
    , "While"
    , "XlaIf"
    , "XlaLaunch"
    , "XlaReduce"
    , "XlaReduceWindow"
    , "XlaSelectAndScatter"
    , "XlaWhile"
    , "_If"
    , "_TPUReplicate"
    , "_While"
    , "_XlaCompile"
    ]

autogenModulesDir :: LocalBuildInfo -> FilePath
#if MIN_VERSION_Cabal(2,0,0)
autogenModulesDir = BuildPaths.autogenPackageModulesDir
#else
autogenModulesDir = BuildPaths.autogenModulesDir
#endif
