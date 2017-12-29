# wav2letter

wav2letter is a simple and efficient end-to-end Automatic Speech
Recognition (ASR) system from Facebook AI Research. The original authors of
this implementation are Ronan Collobert, Christian Puhrsch, Gabriel
Synnaeve, Neil Zeghidour, and Vitaliy Liptchinsky.

wav2letter implements the architecture proposed in
[Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf)
and [Gated ConvNets for Letter-Based ASR](https://arxiv.org/pdf/1609.03193.pdf).

We provide pre-trained models for [Librispeech](http://www.openslr.org/12) dataset.

## Papers

Our approach is detailed in two scientific contributions:
```
@article{collobert:2016,
  author    = {Ronan Collobert and Christian Puhrsch and Gabriel Synnaeve},
  title     = {Wav2Letter: an End-to-End ConvNet-based Speech Recognition System},
  journal   = {CoRR},
  volume    = {abs/1609.03193},
  year      = {2016},
  url       = {http://arxiv.org/abs/1609.03193},
}
```
and
```
@article{liptchinsky:2017,
  author    = {Vitaliy Liptchinsky and Gabriel Synnaeve and Ronan Collobert},
  title     = {Letter-Based Speech Recognition with Gated ConvNets},
  journal   = {ArXiv e-prints},
  volume    = {abs/1712.09444},
  year      = {2017},
  url       = {http://arxiv.org/abs/1712.09444},
}
```

If you use wav2letter or related pre-trained models, then please cite one of these papers.

## Requirements

* A computer running MacOS or Linux.
* [Torch](http://http://torch.ch). We detail in the following how to install it.
* For training on CPU: [Intel MKL](https://software.intel.com/en-us/intel-mkl).
* For training on GPU: [NVIDIA CUDA Toolkit (cuDNN v5.1 for CUDA 8.0)](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* For reading of audio file: [Libsndfile](http://www.mega-nerd.com/libsndfile) - should be available in any standard distribution.
* For standard speech features: [FFTW](http://www.fftw.org) - should be available in any standard distribution.

## Installation

### MKL
If you plan to train on CPU, it is highly recommended to install
[Intel MKL](https://software.intel.com/en-us/mkl).

Update your .bashrc file with the following:
```sh
# We assume Torch will be installed in $HOME/usr.
# Change according to your needs.
export PATH=$HOME/usr/bin:$PATH

# This is to detect MKL during compilation
# but also to make sure it is found at runtime.
INTEL_DIR=/opt/intel/lib/intel64
MKL_DIR=/opt/intel/mkl/lib/intel64
MKL_INC_DIR=/opt/intel/mkl/include

if [ ! -d "$INTEL_DIR" ]; then
    echo "$ warning: INTEL_DIR out of date"
fi
if [ ! -d "$MKL_DIR" ]; then
    echo "$ warning: MKL_DIR out of date"
fi
if [ ! -d "$MKL_INC_DIR" ]; then
    echo "$ warning: MKL_INC_DIR out of date"
fi

# Make sure MKL can be found by Torch.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_DIR:$MKL_DIR
export CMAKE_LIBRARY_PATH=$LD_LIBRARY_PATH
export CMAKE_INCLUDE_PATH=$CMAKE_INCLUDE_PATH:$MKL_INC_DIR
```

### LuaJIT + LuaRocks

The following installs luaJIT and luarocks locally in `$HOME/usr`. If you
want a system-wide installation, remove the
`-DCMAKE_INSTALL_PREFIX=$HOME/usr` option.

```sh
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir build; cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/usr -DWITH_LUAJIT21=OFF
make -j 4
make install
cd ../..
```

In the next sections, we assume `luarocks` and `luajit` are in `$PATH`. If
they are not - and assuming you installed them locally in `$HOME/usr` - you
can instead run `~/usr/bin/luarocks` and `~/usr/bin/luajit`.

### [KenLM Language Model Toolkit](https://kheafield.com/code/kenlm)

If you plan to use the wav2letter decoder, you will need KenLM.

KenLM requires [Boost](http://www.boost.org).
```sh
# make sure boost is installed (with system/thread/test modules)
# actual command might vary depending on your system
sudo apt-get install libboost-dev libboost-system-dev libboost-thread-dev libboost-test-dev
```

Once boost is properly installed, you may install KenLM:
```sh
wget https://kheafield.com/code/kenlm.tar.gz
tar xfvz kenlm.tar.gz
cd kenlm
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/usr -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j 4
make install
cp -a lib/* ~/usr/lib # libs are not installed by default :(
cd ../..
```

### [OpenMPI](https://www.open-mpi.org) and [TorchMPI](https://github.com/facebookresearch/TorchMPI)

If you plan to use multi-CPU/GPUs (and/or multi-machines), you will need
OpenMPI and TorchMPI.

_Disclaimer: it is highly encouraged to recompile OpenMPI yourself. OpenMPI
binaries on standard distributions come with a lot of variance in the
compilation flags. Certain flags are crucial to successfully compile and
run TorchMPI._

First install OpenMPI:
```sh
wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.2.tar.bz2
tar xfj openmpi-2.1.2.tar.bz2
cd openmpi-2.1.2; mkdir build; cd build
./configure --prefix=$HOME/usr --enable-mpi-cxx --enable-shared --with-slurm --enable-mpi-thread-multiple --enable-mpi-ext=affinity,cuda --with-cuda=/public/apps/cuda/9.0
make -j 20 all
make install
```

_Note: works the same with openmpi-3.0.0.tar.bz2, but
--enable-mpi-thread-multiple needs then to be removed._

You may now install TorchMPI:
```sh
MPI_CXX_COMPILER=$HOME/usr/bin/mpicxx ~/usr/bin/luarocks install torchmpi
```

### Torch and other Torch packages
```sh
luarocks install torch
luarocks install cudnn # for GPU support
luarocks install cunn # for GPU support
```

###  wav2letter packages
```
git clone https://github.com/facebookresearch/wav2letter.git
cd wav2letter
cd gtn && luarocks make rocks/gtn-scm-1.rockspec && cd ..
cd speech && luarocks make rocks/speech-scm-1.rockspec && cd ..
cd torchnet-optim && luarocks make rocks/torchnet-optim-scm-1.rockspec && cd ..
cd wav2letter && luarocks make rocks/wav2letter-scm-1.rockspec && cd ..
# Assuming here you got KenLM in $HOME/kenlm
# And only if you plan to use the decoder:
cd beamer && KENLM_INC=$HOME/kenlm luarocks make rocks/beamer-scm-1.rockspec && cd ..
```

## Training wav2letter models

### Data pre-processing

The *data* folder contains a number of scripts for preprocessing various
datasets. For now we provide only LibriSpeech and TIMIT.

Below is an example on how to preprocess LibriSpeech ASR corpus:
```sh
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar xfvz dev-clean.tar.gz
# repeat for train-clean-100, train-clean-360, train-other-500, dev-other, test-clean, test-other
luajit ~/wav2letter/data/librispeech/create.lua ~/LibriSpeech ~/librispeech-proc
luajit ~/wav2letter/data/utils/create-sz.lua librispeech-proc/train-clean-100 librispeech-proc/train-clean-360 librispeech-proc/train-other-500 librispeech-proc/dev-clean librispeech-proc/dev-other librispeech-proc/test-clean librispeech-proc/test-other
```

### Training
```sh
mkdir experiments
luajit ~/wav2letter/train.lua --train -rundir ~/experiments -runname hello_librispeech -arch ~/wav2letter/arch/librispeech-glu-highdropout -lr 0.1 -lrcrit 0.0005 -gpu 1 -linseg 1 -linlr 0 -linlrcrit 0.005 -onorm target -nthread 6 -dictdir ~/librispeech-proc  -datadir ~/librispeech-proc -train train-clean-100+train-clean-360+train-other-500 -valid dev-clean+dev-other -test test-clean+test-other -gpu 1 -sqnorm -mfsc -melfloor 1 -surround "|" -replabel 2 -progress -wnorm -normclamp 0.2 -momentum 0.9 -weightdecay 1e-05
```

### Training on multiple GPUs
Use OpenMPI to spawn multiple training processes, one per GPU:
```sh
mpirun -n 2 --bind-to none  ~/TorchMPI/scripts/wrap.sh luajit ~/wav2letter/train.lua --train -mpi -gpu 1 ...
```
We assume here `mpirun` is in `$PATH`.

## Running the decoder (inference)

We need to do few pre-processing steps to run the decoder.

We first create a dictionary of letters, which includes the special repetition letters we use in wav2letter:
```sh
cat ~/librispeech-proc/letters.lst >> ~/librispeech-proc/letters-rep.lst && echo "1" >> ~/librispeech-proc/letters-rep.lst && echo "2" >> ~/librispeech-proc/letters-rep.lst
```

We then get a language model, and pre-process it. Here, we will use the
[pre-trained language models for LibriSpeech](http://www.openslr.org/11),
but one can also train its own with KenLM. We then pre-process it to
transform words in low caps, and produce their letter transcriptions with
the repetition letters in a particular dictionary `dict.lst`. The script
might warn you about words which are incorrectly transcribed, due to
insufficient number of repetitions letters (here 2, with `-r 2`). This is
not a problem in our case, as these words are rare.
```sh
wget http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz luajit
~/wav2letter/data/utils/convert-arpa.lua ~/3-gram.pruned.3e-7.arpa.gz ~/3-gram.pruned.3e-7.arpa ~/dict.lst -preprocess ~/wav2letter/data/librispeech/preprocess.lua -r 2 -letters letters-rep.lst
```

_Note: one can use the pre-trained 4-gram language model `4-gram.arpa.gz`
instead; pre-processing will take longer._

Optional: subsequent loading of the language model can be made faster by converting it to a binary format
with KenLM (we assume here KenLM is in your `$PATH`).
```sh
build_binary 3-gram.pruned.3e-7.arpa 3-gram.pruned.3e-7.bin
```

We can now generate emissions for a particular trained model, running
`test.lua` on a dataset. The script also displays Letter Error Rate (LER)
and Word Error Rate (WER) - the latter being computed with no
post-processing of the acoustic model.
```sh
luajit ~/wav2letter/test.lua ~/experiments/hello_librispeech/001_model_dev-clean.bin -progress -show -test dev-clean -save
```

Once the emissions are stored, the decoder can be ran to compute the WER
obtained by constraining the decoding with a particular language model:
```sh
luajit ~/wav2letter/decode.lua ~/experiments/hello_librispeech dev-clean -show -letters ~/librispeech-proc/letters-rep.lst  -words ~/dict.lst -lm ~/3-gram.pruned.3e-7.arpa -lmweight 3.1639 -beamsize 25000 -beamscore 40 -nthread 10 -smearing max -show
```

## Pre-trained models
We provide a fully pre-trained model for LibriSpeech:
```
wget https://s3.amazonaws.com/wav2letter/models/librispeech-glu-highdropout.bin
```
NOTE: the model was pre-trained on Facebook infrastructure, so you need to run *test.lua* with slightly different parameters to use it:
```
luajit ~/wav2letter/test.lua ~/librispeech-glu-highdropout.bin -progress -show -test dev-clean -save -datadir ~/librispeech-proc/ -dictdir ~/librispeech-proc/ -gfsai
```

## Join the wav2letter community
* Facebook page: https://www.facebook.com/groups/717232008481207/
* Google group: https://groups.google.com/forum/#!forum/wav2letter-users
* Contact: locronan@fb.com, vitaliy888@fb.com, gab@fb.com

See the [CONTRIBUTING](CONTRIBUTING.md) for how to help out.

## License

See the [LICENSE](LICENSE) as well as the [PATENTS](PATENTS) file.
