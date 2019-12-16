# wav2letter python bindings

wav2letter binding supports ASG criterion (CUDA and CPU backends), featurization of raw audio (MFCC, MFSC, etc.) and beam-search decoder.
Check the [main installation docs](https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings) for build instructions.

After wav2letter package is installed, please, have a look at the examples `examples/` how to use classes and methods of wav2letter from python.
To run examples use the following commands:
- for ASG criterion using CUDA backend `python examples/criterion_example.py`
- for ASG criterion using CPU backend `python examples/criterion_example.py --cpu`
- lexicon beam-search decoder with KenLM word-level language model `python examples/decoder_example.py ../../src/decoder/test`
- featurization `python examples/feature_example.py ../../src/feature/test/data`

[Details on the usage of python bindings](https://github.com/facebookresearch/wav2letter/wiki/Python-bindings)
