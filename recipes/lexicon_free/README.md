# [Who Needs Words? Lexicon-Free Speech Recognition](https://arxiv.org/abs/1904.04479)

## Abstract
Lexicon-free speech recognition naturally deals with the problem of out-of-vocabulary (OOV) words. In this paper, we show that character-based language models (LM) can perform as well as word-based LMs for speech recognition, in word error rates (WER), even without restricting the decoding to a lexicon. We study character-based LMs and show that convolutional LMs can effectively leverage large (character) contexts, which is key for good speech recognition performance downstream. We specifically show that the lexicon-free decoding performance (WER) on utterances with OOV words using character-based LMs is better than lexicon-based decoding, both with character or word-based LMs.

## Reproducing
Acoustic models architectures flags files are provided for each dataset to reproduce results from the paper (training and decoding steps).
Besides this language models training steps are listed to reproduce ngram and ConvLM language models.

Considered benchmarks (acoustic models are char-based):
- Librispeech
- WSJ

Acoustic models are char-based. Decoding step is considered for the following cases:
- word language model
- char language model
- char language model without lexicon


## Dependencies
All dependencies are listed in the `Dockerfile`. To use docker image run
```
sudo docker run --runtime=nvidia --rm -itd --ipc=host --name lexfree wav2letter/wav2letter:lexfree
sudo docker exec -it lexfree bash
```

## Citation
```
@article{likhomanenko2019needs,
  title={Who needs words? lexicon-free speech recognition},
  author={Likhomanenko, Tatiana and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:1904.04479},
  year={2019}
}
```
