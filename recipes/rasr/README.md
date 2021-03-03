# RASR release

This is a repository sharing pre-trained acoustic models and language models for our new paper [Rethinking Evaluation in ASR: Are Our Models Robust Enough?](https://arxiv.org/abs/2010.11745).


## Dependencies

* [`Flashlight`](https://github.com/facebookresearch/flashlight)
* [`Flashlight` ASR app](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr)

## Models

### Acoustic Model

All the acoustic models are retrained using `Flashlight` with [wav2letter++](https://github.com/facebookresearch/wav2letter) consolidated. `Tedlium` is not used as training data here due to license issue. All the training data has more standardized sample rate 16kHz rather than 8kHz used in the paper.

Here, we are releasing models with different architecture and different sizes. Note that the models may not fully reproduce results in the paper because of both data and toolkit implementation discrepancies.

|Achitecture |# Param |Arch File |Path |
| :---: | :---: | :---: | :---: |
|Transformer |300 mil |[am_transformer_ctc_stride3_letters_300Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.arch) |[am_transformer_ctc_stride3_letters_300Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin) |
|Transformer |70 mil |[am_transformer_ctc_stride3_letters_70Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.arch) |[am_transformer_ctc_stride3_letters_70Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin) |
|Conformer |300 mil |[am_conformer_ctc_stride3_letters_300Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_300Mparams.arch) |[am_conformer_ctc_stride3_letters_300Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_300Mparams.bin) |
|Conformer |87 mil |[am_conformer_ctc_stride3_letters_87Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_87Mparams.arch) |[am_conformer_ctc_stride3_letters_87Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_87Mparams.bin) |
|Conformer |28 mil |[am_conformer_ctc_stride3_letters_25Mparams.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams.arch) |[am_conformer_ctc_stride3_letters_25Mparams.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams.bin) |
|Conformer (distillation) |28 mil |[am_conformer_ctc_stride3_letters_25Mparams_distill.arch](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams_distill.arch) |[am_conformer_ctc_stride3_letters_25Mparams_distill.bin](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_conformer_ctc_stride3_letters_25Mparams_distill.bin) |


### Language Model

Language models are trained on Common Crawl corpus as mentioned in paper. We are providing 4-gram LMs with different pruning here with [200k-top words](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_200kvocab.txt). All the LMs are trained with [KenLM toolkit](https://kheafield.com/code/kenlm/).

| Pruning Param |Size (GB) |Path | Arpa Path |
| :---: | :---: | :---: | :---: |
|0 0 5 5 |8.4 |[large](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_large_4gram_prun0-0-5_200kvocab.bin) | - |
|0 6 15 15 |2.5 |[small](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_small_4gram_prun0-6-15_200kvocab.bin)  | [small](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_small_4gram_prun0-6-15_200kvocab.arpa) |

The perplexities of the LMs on different development sets are listed below.

| LM |nov93dev |TL-dev |CV-dev |LS-dev-clean |LS-dev-other |RT03 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Large |313 |158 |243 |303 |304 |227 |
| Small |331 |178 |262 |330 |325 |226 |


### WER

Here we summarize the decoding WER for all releasing models. All the numbers in the table are in format `viterbi WER -> beam search WER (small beam/large beam)`.

|Achitecture |# Param |nov92 |TL-test |CV-test |LS-test-clean |LS-test-other |Hub05-SWB |Hub05-CH |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Transformer |300 mil |3.4 → 2.9/2.9 |7.6 → 5.5/5.4 |15.5 → 11.6/11.2 |3.0 → 3.2/3.2 |7.2 → 6.4/6.4 |6.8 → 6.2/6.2 |11.6 → 10.8/10.7 |
|Transformer |70 mil |4.5 → 3.7/3.5 |9.4 → 6.2/6.1 |19.8 →13.8/13.0 |4 → 3.6 /3.6 |9.7 → 7.7/7.5 |7.5 → 6.6/6.5 |13 → 11.8/11.7 |
|Conformer |300 mil |3.5 → 3.3/3.3 |8.4 → 6.2/6.0 |17 → 12.7/12.0 |3.2 → 3.4/3.4 |8 → 7/6.8 |7 → 6.4/6.5 |11.9 → 10.7/10.5 |
|Conformer |87 mil |4.3 → 3.3/3.3 |8.7 → 6.1/5.9 |18.2 →13.1/12.4 |3.7 → 3.5/3.5 |8.6 → 7.4/7.2 |7.3 → 6.7/6.7 |12.2 → 11.7/11.5 |
|Conformer |28 mil |5 → 3.9/3.8 |10.5 → 6.9/6.6 |22.2 → 15.4/14.4 |4.7 → 4/3.9 |11.1 → 8.9/8.6 |8.8 → 7.8/7.7 |13.7 → 12.4/12.2 |
|Conformer (distillation) |28 mil |4.7 → 3.9/3.8 |9.4 → 6.5/6.4 |19.6 → 14.6/13.8 |4.1 → 3.8/3.8 |9.9 → 8.4/8.2 |7.6 → 6.9/6.8 |13.0 → 12.2/12.0 |

Decoding is done with lexicon-based beam-search decoder using 200k common crawl lexicon and small common crawl lm.
* [tokens](https://[dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt](http://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/tokens.txt))
* [inference lexicon](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lexicon.txt)
* Decoding parameters (`beamthreshold=100, beamsizetoken=30`):

|Achitecture |# Param |LM Weight |Word Score |Beam Size |
| :---: | :---: | :---: | :---: | :---: |
|Transformer |300 mil |1.5 |0 |50/500 |
|Transformer |70 mil |1.7 |0 |50/500 |
|Conformer |300 mil |1.8 |2 |50/500 |
|Conformer |87 mil |2 |0 |50/500 |
|Conformer |28 mil |2 |0 |50/500 |
|Conformer (distilllation) |28 mil |1.4 |0.4 |50/500 |

## Tutorial

To simply serialize all the models and interact with them, please refer to the [`Flashlight` ASR app tutorials](https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr/tutorial).



## Citation

```
@article{likhomanenko2020rethinking,
  title={Rethinking Evaluation in ASR: Are Our Models Robust Enough?},
  author={Likhomanenko, Tatiana and Xu, Qiantong and Pratap, Vineel and Tomasello, Paden and Kahn, Jacob and Avidov, Gilad and Collobert, Ronan and Synnaeve, Gabriel},
  journal={arXiv preprint arXiv:2010.11745},
  year={2020}
}
```
