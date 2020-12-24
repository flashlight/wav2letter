# RASR release

This is a repository sharing pre-trained acoustic models and language models for our new paper [Rethinking Evaluation in ASR: Are Our Models Robust Enough?](https://arxiv.org/abs/2010.11745).


## Dependencies

* [flashlight](https://github.com/facebookresearch/flashlight)

## Models

### Acoustic Model

All the acoustic models are retrained using flashlight with [wav2letter++](https://github.com/facebookresearch/wav2letter) consolidated. `Tedlium` is not used as training data here due to license issue. All the training data has more standardized sample rate 16kHz rather than 8kHz used in the paper.

Here, we are releasing models with different architecture and different sizes. Note that the models may not fully reproduce results in the paper because of both data and toolkit implementation discrepancies.

|Achitecture |# Param |Arch File |Path |
| :---: | :---: | :---: | :---: |
|Transformer |300 mil |[am_transformer_ctc_stride3_letters_300Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.arch) |[am_transformer_ctc_stride3_letters_300Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin) |
|Transformer |70 mil |[am_transformer_ctc_stride3_letters_70Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.arch) |[am_transformer_ctc_stride3_letters_70Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin) |
|Conformer |300 mil |[am_conformer_ctc_stride3_letters_300Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_300Mparams.arch) |[am_conformer_ctc_stride3_letters_300Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_300Mparams.bin) |
|Conformer |87 mil |[am_conformer_ctc_stride3_letters_87Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_87Mparams.arch) |[am_conformer_ctc_stride3_letters_87Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_87Mparams.bin) |
|Conformer |28 mil |[am_conformer_ctc_stride3_letters_25Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams.arch) |[am_conformer_ctc_stride3_letters_25Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams.bin) |



### Language Model

Language models are trained on Common Crawl corpus as mentioned in paper. We are providing 4-gram LMs with different pruning here with [200k-top words](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_200kvocab.txt). All the LMs are trained with [KenLM toolkit](https://kheafield.com/code/kenlm/).

| Pruning Param |Size (GB) |Path |
| :---: | :---: | :---: |
|0 0 5 5 |8.4 |[large](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_large_4gram_prun0-0-5_200kvocab.bin) |
|0 6 15 15 |2.5 |[small](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_small_4gram_prun0-6-15_200kvocab.bin)  |

The perplexities of the LMs on different development sets are listed below.

| LM |nov93dev |TL-dev |CV-dev |LS-dev-clean |LS-dev-other |RT03 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Large |313 |158 |243 |303 |304 |227 |
| Small |331 |178 |262 |330 |325 |226 |


### WER

Here we summarize the decoding WER for all releasing models. All the numbers in the table are in format `viterbi WER -> beam search WER`.

|Achitecture |# Param |nov92 |TL-test |CV-test |LS-test-clean |LS-test-other |Hub05-SWB |Hub05-CH |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Transformer |300 mil |3.4 → 2.9 |7.6 → 5.5 |15.5 → 11.6 |3.0 → 3.2 |7.2 → 6.4 |6.8 |11.6 |
|Transformer |70 mil |4.5 |9.4 |19.8 |4 |9.7 |7.5 |13 |
|Conformer |300 mil |3.5 |8.4 |17 |3.2 |8 |7 |11.9 |
|Conformer |87 mil |4.3 |8.7 |18.2 |3.7 |8.6 |7.3 |12.2 |
|Conformer |28 mil |5 |10.5 |22.2 |4.7 |11.1 |8.8 |13.7 |

Decoding is done with lexicon-based beam-search decoder using 200k common crawl lexicon and small common crawl lm.
* [tokens](https://[dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt](http://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt))
* [inference lexicon](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lexicon.txt)
* Decoding parameters:

|Achitecture |# Param |LM Weight |Word Score |Beam Size |
| :---: | :---: | :---: | :---: | :---: |
|Transformer |300 mil |1.5 |0 |50 |
|Transformer |70 mil | | | |
|Conformer |300 mil | | | |
|Conformer |87 mil | | | |
|Conformer |28 mil |2 |0 |50 |

## Tutorial

To simply serialize all the model and interact with them, please refer to the Flashlight tutorials as in [here](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr/tutorial).



## Citation

```
@article{likhomanenko2020rethinking,
  title={Rethinking Evaluation in ASR: Are Our Models Robust Enough?},
  author={Likhomanenko, Tatiana and Xu, Qiantong and Pratap, Vineel and Tomasello, Paden and Kahn, Jacob and Avidov, Gilad and Collobert, Ronan and Synnaeve, Gabriel},
  journal={arXiv preprint arXiv:2010.11745},
  year={2020}
}
```
