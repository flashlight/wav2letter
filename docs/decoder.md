# Decoder
![Diagram](decoder_sketch.png)

There are two components in decoding — test and decode.

* The test binary is used to compute basic statistics like letter error rate
  (LER) and word error rate (WER) given an acoustic model using the greedy best
  path without the constraint of a language model or lexicon. It will also
  generate an `Emission Set` including the emission matrix as well as other
  target-related information for each sample, so that the `Emission Set` can be
  fed into the decoder directly to generate transcripts without calling the
  models forward function again.
* The decode binary attempts to find the smallest WER using a beam search
  decoder and a language model. It can take as input either an emission set
  generated from the test binary, which enables running hyper-parameter search
  in parallel, or an acoustic model to generate emissions at runtime, which may
  be more convenient when the dataset being decoded is small and a
  hyper-parameter sweep is not required.

Aside from the dataset and acoustic model, two dictionaries must be input to
both binaries.

* Token dictionary: The same dictionary used to train the acoustic model and
  the order of the tokens should also be identical. Each line contains a unique
  token.
* Lexicon: The set of allowed words and their possible spellings used by the
  decoder. Each line is a word and spelling pair that are separated by a tab.
  The spelling is represented by a space-separated sequence of tokens. An
  example entry could be `apple	a p p l e |`, where the pipe `|` at the end is
  the delimiter used during acoustic model training. Note that the same word may
  have multiple spellings; these should be on separate lines.

### Running the `Test`
The dictionaries are specified through the flags `tokens` and `lexicon`. We
also have to set the flags `am` and `emission_dir` to the path of the acoustic
model and the directory where we want to save the emission set. The flags
`datadir` and `test` are combined to specify the datasets we want to run an
experiment on.  Note that we can test on more than 1 dataset, they must be in
the same `datadir` and are specified as a comma-separated list to `test`.

```
<test_cpp_binary> \
-tokens <path/to/tokens.txt> \
-lexicon <path/to/words.txt> \
-am <path/to/acoustic_model.bin> \
-emission_dir <path/to/emission_dir/> \
-datadir <path/to/dataset/> \
-test <path/to/testset/> \
-maxload -1 \
-show
```

### Running the `Decode`
The decoder can take either an acoustic model or an emission set as input but
not both. E.g. only one of the flags `am` and `emission_dir` can be set. In
general, flags across `Decode` and `Test` have similar functions. All the
hyper-parameters are self-documented and can be set accordingly. The flag
`sclite` specifies the path to save the logs, including the *stdout* log and
the hypotheses and references in *sclite* format ([trn](
http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#trn_fmt_name_0)).

#### Using acoustic model
```
<decode_cpp_binary> \
-tokens <path/to/tokens.txt> \
-lexicon <path/to/words.txt> \
-am <path/to/acoustic_model.bin> \
-lm <path/to/language_model.bin> \
-datadir <path/to/dataset/> \
-test <path/to/testset/> \
-sclite <path/to/save/logs/> \
-lmweight 4 \
-wordscore 2.2 \
-maxload 50 \
-beamsize 2500 \
-beamscore 100 \
-silweight -1 \
-nthread_decoder 8 \
-smearing max \
-show \
-showletters
```

#### Using emission set
```
<decode_cpp_binary> \
-tokens <path/to/tokens.txt> \
-lexicon <path/to/words.txt> \
-emission_dir <path/to/emission_dir/> \
-lm <path/to/language_model.bin> \
-datadir <path/to/dataset/> \
-test <path/to/testset/> \
-sclite <path/to/save/logs/> \
-lmweight 4 \
-wordscore 2.2 \
-maxload 50 \
-beamsize 2500 \
-beamscore 100 \
-silweight -1 \
-nthread_decoder 8 \
-smearing max \
-show \
-showletters
```
