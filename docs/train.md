# Train

At its simplest, training a model can be invoked with

```
<train_cpp_binary> train --flagsfile=<path_to_flags>
```

The flags to the train binary can be passed in a flagfile (see this example
[flags file](../recipes/timit/configs/conv_relu/train.cfg)) or as flags on the
command line:

```
<train_cpp_binary> [train|continue|fork] \
--datadir <path/to/data/> \
--tokensdir <path/to/tokens/file/> \
--archdir <path/to/architecture/files/> \
--rundir <path/to/save/models/> \
--arch   <name_of_architecture.arch> \
--train  <train/datasets/> \
--valid  <validation/datasets> \
--lr=0.0001 \
--lrcrit=0.0001
```

### Modes

Training supports three modes: 

- `train` : Train a model from scratch on the given training data.
- `continue` : Continue training a saved model. This can be used for example to
  fine-tune with a smaller learning rate. The `continue` option makes a best
  effort to resume training from the most recent checkpoint of a given model as
  if there were no interruptions.
- `fork` : Create and train a new model from a saved model. This can be used
  for example to adapt a saved model to a new dataset.

### Flags

We give a short description of some of the more important flags here. A
complete list of the flag definitions and short descriptions of their meaning
can be found [here](../src/common/Defines.cpp).

The `datadir` flag is the base path to where all the `train` and `valid`
dataset directories live. Every `train` path will be prefixed by `datadir`.
Multiple datasets can be passed to `train` and `valid` as a comma-separated
list.

Similarly, the `archdir` and `tokensdir` are (optional) base paths to where the
`arch` and `token` files live. For example, the complete architecture file path
will be `<archdir>/<arch>`.

The `rundir` flag is the base directory where the model will be saved and the
`runname` is the subdirectory that will be created to save the model and
training logs. If `runname` is unspecified a directory name based on the date,
time and user will be created.

Most of the training hyperparameter flags have default values. Many of these
you will not need to change. Some of the more important ones include:

```
- `lr` : The learning rate for the model parameters.
- `lrcrit` : The learning rate for the criterion parameters.
- `criterion` : Which criterion (e.g. loss function) to use. Options include `ctc`,
  `asg` or `seq2seq`.
- `batchsize` : The size of the minibatch to use per GPU.
- `maxgradnorm` : Clip the norm of gradient of the model and criterion parameters
  to this value. NB the norm is computed and clipped on the aggregated model
  and criterion parameters.
```


## Distributed

wav2letter++ supports distributed training on multiple GPUs out of the box. To
run on multiple GPUs set pass the flag `-enable_distributed true` and run with
MPI:

```
mpirun -n 8 <train_cpp_binary> [train|continue|fork] \
-enable_distributed true \
<... other flags ..>
```

The above command will run data parallel training with 8 processes (e.g. on 8
GPUs).
