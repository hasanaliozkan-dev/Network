import Data.Matrix
import System.Random


data DenseLayer = DenseLayer { weights :: IO (Matrix Double)
                             , biases :: IO Double}



randomMatrix :: Int -> Int -> IO (Matrix Double)
randomMatrix row col = do
    gen <- newStdGen
    let random_values = take (row*col) $ randomRs (0, 1) gen :: [Double]
    return $ fromList row col random_values

randomNumber :: IO Double
randomNumber = do
    gen <- newStdGen
    let random_value = head $ randomRs (0, 1) gen :: Double
    return random_value

dense :: Int -> Int -> DenseLayer
dense input_size output_size = DenseLayer (randomMatrix output_size input_size) randomNumber  




--Define the dot product of two matrix
dot :: Matrix Double -> Matrix Double -> Matrix Double
dot a b = multStd2 a b


printDense :: DenseLayer -> IO ()
printDense (DenseLayer weights biases) = do
    w <- weights
    b <- biases
    print w
    print b


