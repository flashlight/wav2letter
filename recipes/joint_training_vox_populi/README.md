# VoxPopuli : Wav2letter checkpoints

Wav2letter checkpoints from the [voxpopuli paper](https://arxiv.org/abs/2101.00390) as well as some code to load them. They correspond to the implementation of wav2vec as described in https://arxiv.org/abs/2011.00093.

The VoxPopuli dataset can be directly downloaded from the [VoxPopuli repository](https://github.com/facebookresearch/voxpopuli/).

The code included in this folder is a patched version of the original code developped by
[Chaitanya Talnikar](https://scholar.google.com/citations?user=aHLUKlQAAAAJ) from the [wav2letter team](https://github.com/facebookresearch/wav2letter/tree/masked_cpc/recipes/joint_training) in order to include the pre-training.

## Flashlight version

You can always refer to this flahslight commit for the stable release https://github.com/flashlight/flashlight/commit/8f7af9ec1188bfd7050c47abfac528d21650890f

## Loading the checkpoint

Wav2letter small wav2vec model : https://dl.fbaipublicfiles.com/voxpopuli/vox_populi_100k_500iters.tar.gz
[Depreciated checkpoint https://dl.fbaipublicfiles.com/voxpopuli/wav2letter_100k_small.tar.gz]

Our checkpoint is using fl::ext::Serializer. The items are saved in the following order:

```
filename,
FL_APP_ASR_VERSION, // std::string
config,             // std::unordered_map<std::string, std::string>
network,            // fl::Sequential
criterion,          // CPCCriterion (Subclass of fl::app::asr::SequenceCriterion) : unsupervised CPC criterion
criterion2,         // fl::app::asr::SequenceCriterion : supervised CTC criterion
netoptim,           // fl::FirstOrderOptimizer : Optimizer for the unsupervised loss (adam)
netoptim2,          // fl::FirstOrderOptimizer : Optimizer for the supervised loss (adam)
critoptim,          // fl::FirstOrderOptimizer
critoptim2          // fl::FirstOrderOptimizer
```

The network consists in a base feature network topped with a classifier.
To use it for fine-tuning, you need to load the network without its last layer:

```
void LoadFeatures(std::shared_ptr<fl::Sequential>  net0, std::shared_ptr<fl::Sequential> net){

    auto modules_0 = net0->modules();
    int n_layers = modules_0.size() - 1
    for (int i =0; i< n_layers; i++){
        net->add(modules_0[i]);
    }
}
```


## Building the Common Voice manifest files

First, download the datasets you're interested in from the [Common Voice website](https://commonvoice.mozilla.org/en/datasets).
Uncompress the data and copy them into $COMMON_VOICE_DIR/$LANG. You should get the following structure:
```
[COMMON_VOICE_DIR]
├──[LANG]/
    ├── clips/
    │   ├── *.mp3 files
    |__ dev.tsv
    |__ invalidated.tsv
    |__ other.tsv
    |__ test.tsv
    |__ train.tsv
    |__ validated.tsv
    |__ reported.tsv (as of Corpus 5.0)
```

Then run the following command:
```
export COMMON_VOICE_DIR= path to the common voice directory described above
cd prepare_data
bash build_cc_data.sh $LANG
```
The script will produce the manifest files associated with the validated, train, dev and test sets. As well as the lexicon and the token files.

## Fine-tuning the model

A training script is available is the scripts folder. To use it run:
```
export COMMON_VOICE_DIR= path to the common voice directory described above
export WAV2LETTERDIR= path to wav2letter root directory
bash train_lang.sh $DIR_CHECKPOINT $LANG
```

Where DIR_CHECKPOINT is the directory where you have uncomprssed the checkpoint and $LANG is the language you want to train your model on.

## Decoder

A decoding script is also available. It will run the decoding on the dev subset.
```
export COMMON_VOICE_DIR= path to the common voice directory described above
export WAV2LETTERDIR= path to wav2letter root directory
bash decode_lang.sh $DIR_CHECKPOINT $LANG
```

## Results

Performances on CommonVoices without language model (old version: checking the non-regression):

| Language        | Fine-tuning size |                 Dev      |       Test        |
| --------------- |:----------------:|:------------------------:|:-----------------:|
| De              | 314h             | CER 3.83 WER: 15.0       | CER 4.70 WER: 17.0|
| Es              | 203h             | CER 3.49 WER: 10.7       | CER 4.04 WER: 11.9|
| Fr              | 364h             | CER 4.9 WER: 16.9        | CER 5.89 WER: 18.8|

Performances on CommonVoices using a language model built out from CommonVoices data (excluding dev / test):

| Language        | Fine-tuning size |                 Dev      |       Test        |
| --------------- |:----------------:|:------------------------:|:-----------------:|
| De              | 314h             | CER 2.36 WER: 6.76       | CER 2.98 WER: 7.82|
| Es              | 203h             | CER 3.11 WER: 8.93       | CER 3.60 WER: 10.0|
| Fr              | 364h             | CER 2.73 WER: 8.31       | CER 3.57 WER: 9.56|

## Pretrain a model

To pretrain a model by yourself you can run ```sh_voxpopuli/pretrain.sh```.
First, prepare an .lst files listing all of the audio sequences you intend to use for pretraining, let's call it ```unlabelled.lst```.
In this file, each sequence can be given an arbitrary transcription, it doesn't matter for unsupervised training.
For example, ```unlabelled.lst``` could look like this:
```
ID0 PATH_SEQUENCE0 SIZE_0_MS i love potatoes
ID1 PATH_SEQUENCE1 SIZE_1_MS i love potatoes
ID2 PATH_SEQUENCE2 SIZE_2_MS i love potatoes
ID3 PATH_SEQUENCE3 SIZE_3_MS i love potatoes
ID4 PATH_SEQUENCE4 SIZE_4_MS i love potatoes
ID5 PATH_SEQUENCE5 SIZE_5_MS i love potatoes
```

You will also need to provdide the script with a validation set, and token file and a valid lexicon.
If you are running the pre-training fully unsupervised (default option) the kind of tokens and the lexicon don't matter, you just need to provide valid files for wav2letter.

You can also add some supervision to the pretraining procedure, as shown in  https://arxiv.org/abs/2011.00093.
In this case you will need to build a different .lst file with labelled data and make sure that your lexicon and token files are appropriate.

See ```sh_voxpopuli/pretrain.sh``` for more details.

## Citation

```
@misc{wang2021voxpopuli,
      title={VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
      author={Changhan Wang and Morgane Rivière and Ann Lee and Anne Wu and Chaitanya Talnikar and Daniel Haziza and Mary Williamson and Juan Pino and Emmanuel Dupoux},
      year={2021},
      eprint={2101.00390},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
