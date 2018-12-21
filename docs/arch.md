# Writing Architecture Files

wav2letter++ provides a simple way to create `fl::Sequential` module for the acoustic model from text files. These are specified using  the gflags `-arch` and `-archdir`.

Example architecture file:
```
# Comments like this are ignored
V -1 1 NFEAT 0
C1 NFEAT 300 48 2 39
R
C1 300 300 32 1
R
RO 2 0 3 1
L 300 NLABEL
```

While parsing, we ignore lines stating with `#` as comments. We also replace the following tokens `NFEAT` = input feature size (e.g. number of frequency bins), `NLABEL` = output size (e.g. number of grapheme tokens)

The first token in each line represents a specific flashlight module followed by the specification of its parameters.

Here, we describe how to specify different flashlight modules in the architecture files.

**fl::Conv2D** `C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding] [yPadding]`  

*(Use padding `= -1` for `fl::PaddingMode::SAME`)* <br/>

**fl::Linear** `L [inputChannels] [outputChannels]` <br/>

**fl::BatchNorm** `BN [firstDim] [secondDim <OPTIONAL>] [thirdDim <OPTIONAL>]` <br/>

**fl::LayerNorm** `LN [firstDim] [secondDim <OPTIONAL>] [thirdDim <OPTIONAL>]` <br/>

**fl::WeightNorm** `WN [normDim] [Layer]` <br/>

**fl::Dropout** `DO [dropProb]` <br/>

**fl::Pool2D**
   1. Average : `A [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding] [yPadding]`
   1. Max : `M [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding] [yPadding]`

*(Use padding `= -1` for `fl::PaddingMode::SAME`)* <br/>

**fl::View** `V [firstDim] [secondDim] [thirdDim] [fourthDim]`

*(Use `-1` to infer dimension, only one param can be a `-1`. Use `0` to use the corresponding input dimension.)* <br/>

**fl::Reorder** `RO [firstDim] [secondDim] [thirdDim] [fourthDim]` <br/>

**fl::ELU** `ELU` <br/>

**fl::ReLU** `R`  <br/>

**fl::PReLU** `PR [numElements <OPTIONAL>] [initValue <OPTIONAL>]`  <br/>

**fl::Log** `LG` <br/>

**fl::HardTanh** `HT`  <br/>

**fl::Tanh** `T` <br/>

**fl::GatedLinearUnit** `GLU [sliceDim]`  <br/>

**fl::LogSoftmax** `LSM [normDim]` <br/>

**fl::RNN**
   1. RNN : `RNN [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]`
   1. GRU : `GRU [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]`
   1. LSTM : `LSTM [inputSize] [outputSize] [numLayers] [isBidirectional] [dropProb]`
