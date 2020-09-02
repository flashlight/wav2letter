# Beam candidates dumping and rescoring

Here we describe how to dump the beam candidates and perform rescoring with GCNN and Transformer LMs we did in the paper.

## Dependencies
- [fairseq](https://github.com/pytorch/fairseq), commit `d80ad54`: put the `forward_lm.py` into fairseq directory (otherwise doesn't work with fairseq import)

## Beam candidates dumping
- Fix the paths inside `decode*.cfg`
- Run beam dumping with `decode*.cfg` for each model:
  - CTC Librispeech Transformer, ngram and GCNN decoding
  - S2S Librispeech Transformer, ngram and GCNN decoding
  - CTC Librivox Transformer, ngram and GCNN decoding
  - S2S Librivox Transformer, ngram and GCNN decoding
```
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/other/config --minloglevel=0 --logtostderr=1 --emission_dir='' --test=dev-other.lst
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/other/config --minloglevel=0 --logtostderr=1 --emission_dir='' --test=test-other.lst
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/clean/config --minloglevel=0 --logtostderr=1 --emission_dir='' --test=dev-clean.lst
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/clean/config --minloglevel=0 --logtostderr=1 --emission_dir='' --test=test-clean.lst
```
Also run specific diverse beam dump for s2s Librispeech with gcnn decoding (the best one) to have diverse beam (optimization of beam search decoder is done with `--beamsize=50 --beamthreshold==10 --beamsizetoken=10`):
- please copy files from `src/*` to the `wav2letter/src/libraries/decoder` to overwrite Lexicon-free s2s decoder algorithm
- rebuild Decode.cpp with this updates
- run the following beam dump
```
[...]/wav2letter/build/Decoder --flagsfile decode_transformer_s2s_gcnn_other_ls_completed_hyps.cfg --minloglevel=0 --logtostderr=1 --emission_dir='' --test=dev-other.lst
[...]/wav2letter/build/Decoder --flagsfile decode_transformer_s2s_gcnn_other_ls_completed_hyps.cfg --minloglevel=0 --logtostderr=1 --emission_dir='' --test=test_other.lst
```

## Generate perplexity for each candidate in the beam
We use word-based GCNN and word-based Transformer to rescore, so at first we generate their perplexities (actually it is loss for the sentecnce) for each candidate in the beam
```
cd [FAIRSEQ]
# convlm
python forward_lm.py --model [MODEL_PATH]/checkpoint_best.pt --dict [MODEL_DATA_PATH]/dict.txt --text [BEAM DUMP].lst.hyp --out convlm.ppl --model-type convlm --max-tokens 1024 --skip 1
# transformer
python forward_lm.py --model [MODEL_PATH]/checkpoint_best.pt --dict [MODEL_DATA_PATH]/dict.txt --text [BEAM DUMP].lst.hyp --out transformer.ppl --model-type transformer --max-tokens 1024 --skip 1
```

## Rescoring
- running random search for S2S models
```
# for name in dev-other dev-clean
python rescore.py --hyp "[PATH_TO_$name].hyp" --list "[DATA_DST]/lists/$name.lst" --convlm="[PATH]/convlm_$name.ppl" --tr="[PATH]/transformer_$name.ppl" --in_wts=0,0,0 --search
# then eval found best weight for test-other and test-clean
python rescore.py --hyp "[PATH_TO_$name].hyp" --list "[DATA_DST]/lists/$name.lst" --convlm="[PATH]/convlm_$name.ppl" --tr="[PATH]/transformer_$name.ppl" --in_wts=w1,w2,w3
```
- running grid search for CTC models (for ngram LM add also `--top=large` to eval 2500 beam)
```
# for name in dev-other dev-clean
python rescore.py --hyp "[PATH_TO_$name].hyp" --list "[DATA_DST]/lists/$name.lst" --convlm="[PATH]/convlm_$name.ppl" --tr="[PATH]/transformer_$name.ppl" --in_wts=0,0,0 --search --gridsearch
# then eval found best weight for test-other and test-clean
python rescore.py --hyp "[PATH_TO_$name].hyp" --list "[DATA_DST]/lists/$name.lst" --convlm="[PATH]/convlm_$name.ppl" --tr="[PATH]/transformer_$name.ppl" --in_wts=w1,w2,w3
```

### Optimal weights of rescoring for Librispeech models (tr LM, GCNN lm, transcritpion len):
- CTC ngram
  - clean 0.4,0.2,0.1
  - other 0.8,0,0.5
- CTC gcnn
  - clean 0.5,0.1,0.2
  - other 0.8,0,0.5
- S2S ngram
  - clean 0.24980175230211288,0.04919538965148296,0.27051900934773476 (top-3)
  - other 0.43353711959171454,0.02376409689162373,0.36902974241819764 (top-40)
- S2S gcnn
  - clean 0.44292736054463794,0,0.5589561526817741 (top-3)
  - other 0.8121556103144534,-0.13482344019156423,0.5317810935135496 (top-6)
  - other (with completed hyps) 0.8530679672776722,-0.351434501804079,0.5151134574867755 (top-10)

### Optimal weights of rescoring for Librivox models (tr LM, GCNN lm, transcritpion len):
- CTC ngram
  - clean 0.5,0,0.4
  - other 0.4,0,0.3
- CTC gcnn
  - clean 0.4,0,0.2
  - other 0.4,0,0.3
- S2S ngram
  - clean 0.24980175230211288,0.04919538965148296,0.27051900934773476 (top-3)
  - other 0.510874298897509,-0.1756293581052848,0.39350304055842433 (top-50)
- S2S gcnn
  - clean 0.4769702057546221,-0.26001972366117654,0.3727554580971921 (top-250)
  - other 0.459099594595537,0.038747686960546535,0.33754803186788784 (top-9)
