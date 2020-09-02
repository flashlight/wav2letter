# [Learning Filterbanks from Raw Speech for Phone Recognition)](https://arxiv.org/pdf/1711.01161.pdf)


We currently provide recipe for the baseline model on TIMIT used in the paper.
At first, prepare data for training (set the paths instead of `[...]`, `[DATA_DST]` and `[MODEL_DST]`)
```
python prepare.py \
  --src [...]/timit \
  --data_dst [DATA_DST] \
  --model_dst [MODEL_DST] \
  --sph2pipe [...]/sph2pipe_v2.5/sph2pipe
```

Besides TIMIT data the auxiliary files for acoustic model training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── lexicon.txt
│   └── tokens.txt
```

To train the baseline model run (Set the full path to wav2letter for `[...]`).
```
[...]/wav2letter/build/Train train --flagsfile train_baseline_conv_relu.cfg --minloglevel=0 --logtostderr=1

```

## Citation
```
@inproceedings{zeghidour2018learning,
  title={Learning filterbanks from raw speech for phone recognition},
  author={Zeghidour, Neil and Usunier, Nicolas and Kokkinos, Iasonas and Schaiz, Thomas and Synnaeve, Gabriel and Dupoux, Emmanuel},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5509--5513},
  year={2018},
  organization={IEEE}
}
```
