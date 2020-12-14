# Tools

## Streaming TDS model conversion for running inference pipeline
Once a model is trained in wav2letter++ for streaming TDS models using the [provided recipe](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/streaming_convnets) possibly customized to suit ones' use-case, the model needs to be serialized to a format which wav2letter@anywhere inference platform can load. `StreamingTDSModelConverter` can be used to do this. Note that the script only supports models trained using the streaming TDS + CTC style architectures as described in the paper [here](https://research.fb.com/publications/scaling-up-online-speech-recognition-using-convnets/).
### Using the Pipeline
Build the tool with `make streaming_tds_model_converter`.
And to run the binary:
```
[path to binary]/streaming_tds_model_converter \
    -am [path to model] \
    --outdir [output directory]
```
The output directory will contain
- `tokens.txt` - Tokens file (with blank symbol included)
- `acoustic_model.bin` - Serialized acoutic model
- `feature_extractor.bin` - Serialized feature extraction model which perform log-mel feature extraction and local normalization

These files can be used to run inference on audio files along with a few other files required for decoding like language model, lexicon etc. See the [tutorial](https://github.com/facebookresearch/wav2letter/wiki/Inference-Run-Examples) for more details.
