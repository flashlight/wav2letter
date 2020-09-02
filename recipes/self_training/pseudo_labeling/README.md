# Generating, Analyzing, and Filtering Pseudo-Labels

Tools used to analyize and generate pseudo-labels used in experiments can be found here. Refer to [the paper](https://arxiv.org/abs/1909.09116) for optimal parameters used to generate results from the paper.

## Generating Labels

Using raw sclite logs from decoding, pseudo-labels can be generated and filtering applied by running:

```
python3 generate_synthetic_data.py \
    --input [path to decoder stdout] \
    --listpath [path of listfile containing ground truth] \
    --output [path to new output list file]
    # Whether or not to perform any sort of filtering
    --filter \
    # Whether or not to filter EOS warnings
    --warnings \
    # Whether or not to do ngram filtering
    --ngram \
    # If ngram filtering is enabled, the number of times an n-gram must appear to be filtered
    --ngram_appearance_threshold \
    # The size of the n-gram to appear
    --ngram_size \
    # If doing filtering show the results
    --print_filtered_results
```

## Analyzing Pseudo-Labels

To run basic analysis, including the Oracle word error rate, the number of samples, and the duration of a dataset, use the tools provided in `AnalyzeDataset`. Make sure `W2L_BUILD_RECIPES` is `ON` in CMake when building wav2letter.

Run:
```
./analyze_pseudo_label_dataset \
     --infile [path to generated pseudo-label lst file] \
     --groundtruthfile [path to ground truth lst file]
```

## Generating a Synthetic Leixcon

To generate a synthetic lexicon from decoder output, run:

```
python3 generate_synthetic_lexicon.py \
    --inputhyp=[path to decoder sclite log output] \
    --inputlexicon=[path to an existing lexicon, e.g. train-clean-100] \
    --output=[output file]
```

Lexicons can also be combined with other lexicons by running the following:

```
python3 combine_synthetic_lexicon.py \
    --lexicon1=[train-clean-100 lexicon] \
    --lexicon2=[synthetic lexicon] \
    --output=[output file]
```

Note that the resulting lexicon will have words ordered
