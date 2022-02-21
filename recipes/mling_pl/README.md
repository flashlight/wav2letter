# Pseudo Labeling for Massively Multilingual ASR

Semi-supervised learning through pseudo-labeling has become a staple of state-of-the-art monolingual speech recognition systems. In this work, we extend pseudo-labeling to massively multilingual speech recognition with 60 languages. We propose a simple pseudo-labeling recipe that works well even with low-resource languages: train a supervised multilingual model, fine-tune it with semi-supervised learning on a target language, generate pseudo-labels for that language, and train a final model using pseudo-labels for all languages, either from scratch or by fine-tuning. Experiments on the labeled Common Voice and unlabeled VoxPopuli datasets show that our recipe can yield a model with better performance for many languages that also transfers well to LibriSpeech.

We provide are pretrained models and a script to run inference on a sample audio file.

## Inference

#### Step 1:
Download the pretrained model and tokens file

| Model | Arch | Link |
| - | - | - |
Large | model_with_externally_controlled_reshaping_big_lid.cpp | https://dl.fbaipublicfiles.com/wav2letter/mling_pl/checkpoint_large.bin

Tokens file : https://dl.fbaipublicfiles.com/wav2letter/mling_pl/tokens-all.lst

#### Step 2:

Install flashlight - https://github.com/flashlight/flashlight with ASR app flag `FL_BUILD_APP_ASR=ON`. Use the commit id `8f7af9ec1188bfd7050c47abfac528d21650890f` .

#### Step 3:
Prepare a file with the list of audio files in this format
```
0 <path_to_file1> <duration1>
1 <path_to_file2> <duration2>
2 <path_to_file3> <duration3>
```

#### Step 4:

Run inference using the following command from flashlight build directory

```
bin/asr/fl_asr_test \
    --test <audio_file_list> \
    --am <path_to_model_checkpoint.bin> \
    --arch <path_to_model_arch.so> \
    --tokens <path_to_tokens_file/tokens-all.lst> \
    --datadir ''  \
    --emission_dir ''  \
    --show
```

To compile `*.cpp` architectures into `*.so` use cmake/make command in flashlight and provide `-DFL_PLUGIN_MODULE_SRC_PATH=path/to/*.cpp` flag.


## Citation
```
@article{lugosch2021pseudo,
  title={Pseudo-Labeling for Massively Multilingual Speech Recognition},
  author={Lugosch, Loren and Likhomanenko, Tatiana and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:2111.00161},
  year={2021}
}
```
