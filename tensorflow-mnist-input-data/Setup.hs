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

{-# LANGUAGE LambdaCase #-}

-- | Downloads the MNIST data set and packages them as data files.
module Main where

import Control.Monad (when)
import Data.Maybe (fromMaybe)
import Distribution.PackageDescription
    ( GenericPackageDescription(packageDescription)
    , dataDir
    )
import Distribution.Simple
    ( UserHooks(..)
    , defaultMainWithHooks
    , simpleUserHooks
    )
import System.IO (hPutStrLn, stderr)
import System.FilePath ((</>))
import System.Directory (doesFileExist)
import qualified Crypto.Hash as Hash
import qualified Data.ByteString.Lazy as B
import qualified Network.HTTP as HTTP
import qualified Network.Browser as Browser
import qualified Network.URI as URI

main :: IO ()
main = defaultMainWithHooks downloadingDataFiles

downloadingDataFiles :: UserHooks
downloadingDataFiles = hooks
    { confHook = \gh@(g, _) c -> downloadFiles g >> confHook hooks gh c
    }
  where
    hooks = simpleUserHooks
    downloadFiles :: GenericPackageDescription -> IO ()
    downloadFiles g = do
        let dir = dataDir (packageDescription g)
        mapM_ (maybeDownload dir) fileInfos

maybeDownload :: FilePath -> (String, String) -> IO ()
maybeDownload dataDir (basename, sha256) = do
    let filePath = dataDir </> basename
    exists <- doesFileExist filePath
    when (not exists) $ do
        let url = urlPrefix ++ basename
        hPutStrLn stderr ("Downloading " ++ url)
        httpDownload url filePath
    verify filePath sha256

httpDownload :: String -> FilePath -> IO ()
httpDownload url outFile = do
    let uri = fromMaybe
              (error ("Can't be: invalid URI " ++ url))
              (URI.parseURI url)
    (_, rsp)
        <- Browser.browse $ do
              Browser.setAllowRedirects True
              Browser.setCheckForProxy True
              Browser.request $ HTTP.defaultGETRequest_ uri
    case HTTP.rspCode rsp of
        (2, 0, 0) -> B.writeFile outFile $ HTTP.rspBody rsp
        s -> error ( "Failed to download " ++ url ++ " error code " ++ show s
                     ++ helpfulMessage
                    )

verify :: FilePath -> String -> IO ()
verify filePath hash = do
    let sha256 = Hash.hashlazy :: B.ByteString -> Hash.Digest Hash.SHA256
    computed <- show . sha256 <$> B.readFile filePath
    when (hash /= computed) $
        error ( "Incorrect checksum for " ++ filePath
                 ++ "\nexpected " ++ hash
                 ++ "\ncomputed " ++ computed
                 ++ helpfulMessage
              )

urlPrefix = "http://yann.lecun.com/exdb/mnist/"

-- | File names relative to 'urlPrefix' and their sha256.
fileInfos = [
    ( "train-images-idx3-ubyte.gz"
    , "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
    )
    ,
    ( "train-labels-idx1-ubyte.gz"
    , "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
    )
    ,
    ( "t10k-images-idx3-ubyte.gz"
    , "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
    )
    ,
    ( "t10k-labels-idx1-ubyte.gz"
    , "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
    )
    ]

helpfulMessage =
    unlines
        ( ""
        : ""
        : "Please download the following URLs manually and put them in data/"
        : [ urlPrefix ++ h | (h, _) <- fileInfos ]
        )
