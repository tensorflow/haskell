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

-- | Generates the wrappers for Ops shipped with tensorflow.
module Main where

import Distribution.Simple.BuildPaths (autogenModulesDir)
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

blackList =
    [ -- Requires the "func" type:
      "SymbolicGradient"
      -- Easy: support larger result tuples.
    , "ParseSingleSequenceExample"
    , "Skipgram"
    ]
