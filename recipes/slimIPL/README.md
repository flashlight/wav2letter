# slimIPL

Recent results in end-to-end automatic speech recognition have demonstrated the efficacy of pseudo-labeling for semi-supervised models trained both with Connectionist Temporal Classification (CTC) and Sequence-to-Sequence (seq2seq) losses. Iterative Pseudo-Labeling (IPL), which continuously trains a single model using pseudo-labels iteratively re-generated as the model learns, has been shown to further improve performance in ASR. We improve upon the IPL algorithm: as the model learns, we propose to iteratively re-generate transcriptions with hard labels (the most probable tokens), that is, without a language model. We call this approach Language-Model-Free IPL (slimIPL) and give a resultant training setup for low-resource settings with CTC-based models. slimIPL features a dynamic cache for pseudo-labels which reduces sensitivity to changes in relabeling hyperparameters and results in improved training stability. slimIPL is also highly-efficient and requires 3.5-4x fewer computational resources to converge than other state-of-the-art semi/self-supervised approaches. With only 10 hours of labeled audio, slimIPL is competitive with self-supervised approaches, and is state-of-the-art with 100 hours of labeled audio without the use of a language model both at test time and during pseudo-label generation.

## Training

All models are trained on 16 gpus. To run training
```
[w2l]/build/recipes/slimIPL/src/train_slimipl train \
   --flagsfile=[Put *.cfg file] \
   --rundir=[Path where artifacts of training to save] \
   --enable_distributed=true
```
To compile `*.cpp` architectures into `*.so` use cmake/make command in flashlight and provide `-DFL_PLUGIN_MODULE_SRC_PATH=path/to/*.cpp` flag.

Learning rate decay is done manually as soon as model reaches platueu two times.  When learning rate is decayed the best snapshot is chosen using shallow decoding. Use these additional flags to continue training with decayed learning rate.  Shallow decoding will be running as soon as `--lm` flag is provided. Take best snapshot as `*_model_dev-clean_decoder.bin` and `*_model_dev-other_decoder.bin`:
```
[w2l]/build/recipes/slimIPL/src/train_slimipl continue [model/path] \
   --lr=[decay lr] \
   --lm=[path/to/lm.bin] \ # use the same one as for 4-gram decoding
   --beamsize=50 \
   --beamsizetoken=5 \
   --beamthreshold=1000000 \
   --silscore=0 \
   --wordscore=0 \
   --unkscore=0 \
   --logadd=true \
   --smearing=max \
   --lmweight_low=0 \
   --lmweight_high=4.0 \
   --lmweight_step=0.2 \
```

## Decoding & Beam Dump

General command used for all models is following (only LM weight and word score are different)
```
flashlight/build/bin/asr/fl_asr_decode \
        --lm=[path/to/lm.bin] \
        --am=[path/to/am.bin] \
        --test=lists/[list] \
        --emission_dir= \
        --datadir='' \
        --nthread_decoder=10 \
        --smearing=max \
        --beamsize=1000 \
        --lmweight=$lmweight --wordscore=$wordscore \
        --beamthreshold=100000 \
        --beamsizetoken=100 \
        --decodertype=wrd \
        --lmtype=kenlm \
        --uselexicon=true \
        --unkscore=-inf \
        --logadd=true \
        --lexicon=beam-search.lexicon \
        --sclite=[path/to/dump/beam] \
        --isbeamdump=true \
        --arch=[path/to/arch.so] \
        --nthread_decoder_am_forward=1
```
Best decoding params found on the validation data are following:

| Sup. data | Unsup. data | clean LM weight | clean word score | other LM weight | other word score |
| - | - | - | - | - | - |
10h | - | 3.799 | -1.050 | 3.689 | -1.826
10h | 960h | 5.513 | 2.239 | 3.903 | -2.083
100h | - | 2.708 | -2.312 | 3.183 | -0.721
100h | 860h | 2.064 | -0.143 | 2.137 | -0.575

Language model used for word-based 4-gram decoding is [here](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_word_4g_200kvocab.bin).

## Rescoring
Rescoring exactly follows https://github.com/flashlight/wav2letter/tree/master/recipes/sota/2019/rescoring using only transformer LM scores (no GCNN), Transformer language model can be found [here](https://github.com/flashlight/wav2letter/tree/master/recipes/sota/2019).

Rescoring params found on the validation data are following:

| Sup. data | Unsup. data | clean LM weight | clean Length weight | other LM weight | other Length weight |
| - | - | - | - | - | - |
10h | - | 2.3 | 1.2 | 2.4 | 2.2
10h | 960h | 2.4 | 1.7 | 2.3 | 1.3
100h | - | 1.8 | 1.5 | 2.2 | 1.9
100h | 860h | 1.4 | 1.3 | 1.6 | 0.6

## Ablations

- To run ablations from the paper just change `slimIPL_*` flags to necessary values.
- To run EMA ablation use flags `--slimIPL_ema=true --slimIPL_ema_decay=[0.999]`
- To run "naive" approach (also EMA without dynamic cache) use `--slimIPL_type=naive`

## Citation
```
@article{likhomanenko2020slimipl,
  title={slimIPL: Language-model-free iterative pseudo-labeling},
  author={Likhomanenko, Tatiana and Xu, Qiantong and Kahn, Jacob and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:2010.11524},
  year={2020}
}
```
