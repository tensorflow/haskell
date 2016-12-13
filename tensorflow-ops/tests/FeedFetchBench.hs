-- Disable full-laziness to keep ghc from optimizing most of the benchmark away.
{-# OPTIONS_GHC -fno-full-laziness #-}
import Control.DeepSeq (NFData(rnf))
import Control.Exception (evaluate)
import Control.Monad.IO.Class (liftIO)
import Criterion.Main (defaultMain, bgroup, bench)
import Criterion.Types (Benchmarkable(..))
import qualified Data.Vector.Storable as S
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF

-- | Create 'Benchmarkable' for 'TF.Session'.
--
-- The entire benchmark will be run in a single tensorflow session. The
-- 'TF.Session' argument will be run once and then its result will be run N
-- times.
nfSession :: NFData b => TF.Session (a -> TF.Session b) -> a -> Benchmarkable
nfSession init x = Benchmarkable $ \m -> TF.runSession $ do
    f <- init
    -- Can't use replicateM because n is Int64.
    let go n | n <= 0    = return ()
             | otherwise = f x >>= liftIO . evaluate . rnf >> go (n-1)
    go m

-- | Benchmark feeding and fetching a vector.
feedFetchBenchmark :: TF.Session (S.Vector Float -> TF.Session (S.Vector Float))
feedFetchBenchmark = do
    input <- TF.build (TF.placeholder (TF.Shape [-1]))
    output <- TF.build (TF.render (TF.identity input))
    return $ \v -> do
        let shape = TF.Shape [fromIntegral (S.length v)]
            inputData = TF.encodeTensorData shape v
            feeds = [TF.feed input inputData]
        TF.runWithFeeds feeds output

main :: IO ()
main = defaultMain
    [ bgroup "feedFetch"
        [ bench "4 byte" $ nfSession feedFetchBenchmark (S.replicate 1 0)
        , bench "4 KiB" $ nfSession feedFetchBenchmark (S.replicate 1024 0)
        , bench "4 MiB" $ nfSession feedFetchBenchmark (S.replicate (1024^2) 0)
        ]
    ]
