/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/Defines.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

namespace w2l {

// DATA OPTIONS
DEFINE_string(train, "", "comma-separated list of training data");
DEFINE_string(valid, "", "comma-separated list of valid data");
DEFINE_string(test, "", "comma-separated list of test data");
DEFINE_int64(batchsize, 1, "batch size (per process in distributed training)");
DEFINE_int64(
    validbatchsize,
    -1,
    "batch size (per process in distributed training) for the valid data, if -1 then use train batchsize");
DEFINE_string(input, "flac", "input feature");
DEFINE_int64(samplerate, 16000, "sample rate (Hz)");
DEFINE_int64(channels, 1, "number of input channels");
DEFINE_string(tokens, "tokens.txt", "path/to/tokens");
DEFINE_bool(usewordpiece, false, "use word piece as target");
DEFINE_int64(
    replabel,
    0,
    "replace up to replabel reptitions by additional classes");
DEFINE_string(surround, "", "surround target with provided label");
DEFINE_bool(noresample, false, "do not resample training data");
DEFINE_bool(eostoken, false, "append target with end of sentence token");
DEFINE_string(
    dataorder,
    "input",
    "bin method to use for binning samples, input: in order of length of \
    input, input_spiral: binning using transcript(reference) length , \
    and spiral along audiolength, output_spiral: binning using audio length and \
    spiral along reference lenth");
DEFINE_int64(inputbinsize, 100, "Bin size along audio length axis");
DEFINE_int64(outputbinsize, 5, "Bin size along transcript length axis");
DEFINE_bool(blobdata, false, "use blobs instead of folders as input data");
DEFINE_string(
    wordseparator,
    kSilToken,
    "extra word boundaries to be inserted during target generation");
DEFINE_double(
    sampletarget,
    0.0,
    "probability [0.0, 1.0] for randomly sampling targets from a lexicon if there are multiple mappings from a word");

// FILTERING OPTIONS
DEFINE_int64(minisz, 0, "min input size (in msec) allowed during training");
DEFINE_int64(
    maxisz,
    std::numeric_limits<int64_t>::max(),
    "max input size (in msec) allowed during training");
DEFINE_int64(
    maxtsz,
    std::numeric_limits<int64_t>::max(),
    "max target size allowed during training");
DEFINE_int64(mintsz, 0, "min target size allowed during training");

// NORMALIZATION OPTIONS
DEFINE_int64(localnrmlleftctx, 0, "left context size for local normalization");
DEFINE_int64(
    localnrmlrightctx,
    0,
    "right context size for local normalization");
DEFINE_string(onorm, "none", "output norm (none");
DEFINE_bool(sqnorm, false, "use square-root while normalizing criterion loss");
DEFINE_bool(lrcosine, false, "use cosine learning rate schedule");

// LEARNING HYPER-PARAMETER OPTIONS
DEFINE_int64(iter, std::numeric_limits<int64_t>::max(), "number of updates");
DEFINE_bool(itersave, false, "save model at each iteration");
DEFINE_double(lr, 1.0, "learning rate");
DEFINE_double(momentum, 0.0, "momentum factor");
DEFINE_double(weightdecay, 0.0, "weight decay (L2 penalty)");
DEFINE_double(lrcrit, 0, "criterion learning rate");
DEFINE_int64(warmup, 1, "the LR warmup parameter, in updates");
DEFINE_int64(
    saug_start_update,
    -1,
    "Use SpecAugment starting at the update number inputted. -1 means no SpecAugment");
DEFINE_int64(
    lr_decay,
    std::numeric_limits<int64_t>::max(),
    "Epoch for the first LR decay");
DEFINE_int64(
    lr_decay_step,
    std::numeric_limits<int64_t>::max(),
    "Epochs for each new LR decay");
DEFINE_double(maxgradnorm, 0, "Clip gradients at value (0 = no clipping)");
DEFINE_double(adambeta1, 0.9, "beta1 in the Adam optimizer");
DEFINE_double(adambeta2, 0.999, "beta2 in the Adam optimizer");
DEFINE_double(optimrho, 0.9, "rho in the optimizer");
DEFINE_double(optimepsilon, 1e-8, "epsilon in the optimizer");

// LR-SCHEDULER OPTIONS
DEFINE_int64(
    stepsize,
    std::numeric_limits<int64_t>::max(),
    "We multiply LR by gamma every stepsize updates");
DEFINE_double(gamma, 1.0, "the LR annealing multiplier");

// OPTIMIZER OPTIONS
DEFINE_string(netoptim, kSGDOptimizer, "optimizer for the network");
DEFINE_string(critoptim, kSGDOptimizer, "optimizer for the criterion");

// MFCC OPTIONS
DEFINE_bool(mfcc, false, "use standard htk mfcc features as input");
DEFINE_bool(pow, false, "use standard power spectrum as input");
DEFINE_int64(mfcccoeffs, 13, "number of mfcc coefficients");
DEFINE_bool(mfsc, false, "use standard mfsc features as input");
DEFINE_double(melfloor, 1.0, "specify optional mel floor for mfcc/mfsc/pow");
DEFINE_int64(filterbanks, 40, "Number of mel-filter bank channels");
DEFINE_int64(devwin, 0, "Window length for delta and doubledelta derivatives");
DEFINE_int64(fftcachesize, 1, "number of cached cuFFT plans in GPU memory");
DEFINE_int64(
    framesizems,
    25,
    "Window size in millisecond for power spectrum features");
DEFINE_int64(
    framestridems,
    10,
    "Stride millisecond for power spectrum feature");

// SPECAUGMENT OPTIONS
DEFINE_int64(saug_fmaskf, 27, "Max number of frequency bands that are masked");
DEFINE_int64(saug_fmaskn, 2, "Number of frequency masks");
DEFINE_int64(saug_tmaskt, 100, "Max number of timesteps that are masked");
DEFINE_double(
    saug_tmaskp,
    1.0,
    "Max proportion of the input sequence (1.0 is 100%) that can be masked in time");
DEFINE_int64(saug_tmaskn, 2, "Number of time masks");

// RUN OPTIONS
DEFINE_string(datadir, "", "speech data directory");
DEFINE_string(tokensdir, "", "dictionary directory");
DEFINE_string(rundir, "", "experiment root directory");
DEFINE_string(archdir, "", "arch root directory");
DEFINE_string(flagsfile, "", "File specifying gflags");
DEFINE_string(runname, "", "name of current run");
DEFINE_int64(nthread, 1, "specify number of threads for data parallelization");
DEFINE_string(
    tag,
    "",
    "tag this experiment with a particular name (e.g. 'hypothesis1')");
DEFINE_int64(seed, 0, "Manually specify Arrayfire seed.");
DEFINE_int64(
    memstepsize,
    10 * (1 << 20),
    "Minimum allocation size in bytes per array.");
DEFINE_int64(
    reportiters,
    0,
    "number of iterations after which we will run val and save model, \
    if 0 we only do this at end of epoch ");
DEFINE_double(
    pcttraineval,
    100,
    "percentage of training set (by number of utts) to use for evaluation");

// ARCHITECTURE OPTIONS
DEFINE_string(arch, "default", "network architecture");
DEFINE_string(criterion, kAsgCriterion, "training criterion");
DEFINE_int64(encoderdim, 0, "Dimension of encoded hidden state.");

// Seq2Seq Transformer decoder
DEFINE_int64(
    am_decoder_tr_layers,
    1,
    "s2s transformer decoder: number of layers");
DEFINE_double(am_decoder_tr_dropout, 0.0, "s2s transformer decoder: dropout");
DEFINE_double(
    am_decoder_tr_layerdrop,
    0.0,
    "s2s transformer decoder: layerdrop");

// DECODER OPTIONS

DEFINE_bool(show, false, "show predictions");
DEFINE_bool(showletters, false, "show letter predictions");
DEFINE_bool(logadd, false, "use logadd when merging decoder nodes");
DEFINE_bool(uselexicon, true, "use lexicon in decoding");
DEFINE_bool(isbeamdump, false, "dump the decoding beam");

DEFINE_string(smearing, "none", "none, max or logadd");
DEFINE_string(lmtype, "kenlm", "kenlm, convlm");
DEFINE_string(lexicon, "", "path/to/lexicon.txt");
DEFINE_string(lm_vocab, "", "path/to/lm_vocab.txt");
DEFINE_string(emission_dir, "", "path/to/emission_dir/");
DEFINE_string(lm, "", "path/to/language_model");
DEFINE_string(am, "", "path/to/acoustic_model");
DEFINE_string(sclite, "", "path/to/sclite to be written");
DEFINE_string(decodertype, "wrd", "wrd, tkn");

DEFINE_double(lmweight, 0.0, "language model weight");
DEFINE_double(wordscore, 0.0, "word insertion score");
DEFINE_double(silscore, 0.0, "silence insertion score");
DEFINE_double(
    unkscore,
    -std::numeric_limits<float>::infinity(),
    "unknown word insertion score");
DEFINE_double(eosscore, 0.0, "EOS insertion score");
DEFINE_double(beamthreshold, 25, "beam score threshold");

DEFINE_int32(maxload, -1, "max number of testing examples.");
DEFINE_int32(maxword, -1, "maximum number of words to use");
DEFINE_int32(beamsize, 2500, "max overall beam size");
DEFINE_int32(beamsizetoken, 250000, "max beam for token selection");
DEFINE_int32(nthread_decoder_am_forward, 1, "number of threads for AM forward");
DEFINE_int32(nthread_decoder, 1, "number of threads for decoding");
DEFINE_int32(
    lm_memory,
    5000,
    "total memory size for batch during forward pass ");

DEFINE_int32(emission_queue_size, 3000, "max size of emission queue");

DEFINE_double(
    smoothingtemperature,
    1.0,
    "smoothening the probability distribution in seq2seq decoder");
DEFINE_int32(
    attentionthreshold,
    std::numeric_limits<int>::max(),
    "hard attention limit");

// ASG OPTIONS
DEFINE_int64(linseg, 0, "# of updates of LinSeg to init transitions for ASG");
DEFINE_double(linlr, -1.0, "LinSeg learning rate (if < 0, use lr)");
DEFINE_double(
    linlrcrit,
    -1.0,
    "LinSeg criterion learning rate (if < 0, use lrcrit)");
DEFINE_double(
    transdiag,
    0.0,
    "Initial value along diagonal of ASG transition matrix");

// SEQ2SEQ OPTIONS
DEFINE_int64(maxdecoderoutputlen, 200, "max decoder steps during inference");
DEFINE_int64(
    pctteacherforcing,
    100,
    "Percentage of steps to train using teacher forcing");
DEFINE_string(
    samplingstrategy,
    "rand",
    "Sampling strategy to use when pctteacherforcing < 100. rand or model");
DEFINE_double(
    labelsmooth,
    0.0,
    "Fraction to smooth targets with uniform distribution.");
DEFINE_bool(inputfeeding, false, "feed encoder summary to the decoder RNN");
DEFINE_string(attention, "content", "attention type");
DEFINE_string(attnWindow, "no", "attention window type");
DEFINE_int64(attndim, 0, "Dimension of neural location attention");
DEFINE_int64(
    attnconvchannel,
    0,
    "Number of convolutional channels for location attention");
DEFINE_int64(attnconvkernel, 0, "Kernel width for location attention");
DEFINE_int64(numattnhead, 8, "number of heads for multihead attention");
DEFINE_int64(leftWindowSize, 50, "left median window width");
DEFINE_int64(rightWindowSize, 50, "right median window width");
DEFINE_int64(
    maxsil,
    50,
    "maximum number of leading silence frames for the step window");
DEFINE_int64(
    minsil,
    0,
    "minimum number of leading silence frames for the step window");
DEFINE_double(
    maxrate,
    10,
    "maximum ratio between the transcript and the encoded input lengths for the step window");
DEFINE_double(
    minrate,
    3,
    "minimum ratio between the transcript and the encoded input lengths for the step window");
DEFINE_int64(
    softwoffset,
    10,
    "offset for the soft window center (= offset + step * rate)");
DEFINE_double(
    softwrate,
    5,
    "moving rate for the soft window center (= offset + step * rate)");
DEFINE_double(
    softwstd,
    5,
    "std for the soft window shape (=exp(-(t - center)^2 / (2 * std^2)))");
DEFINE_bool(trainWithWindow, false, "use window in training");
DEFINE_int64(
    pretrainWindow,
    0,
    "use window in training for pretrainWindow in updates");
DEFINE_double(gumbeltemperature, 1.0, "temperature in gumbel softmax");
DEFINE_int64(decoderrnnlayer, 1, "The number of decoder rnn layers.");
DEFINE_int64(decoderattnround, 1, "The number of decoder attention rounds.");
DEFINE_double(decoderdropout, 0.0, "decoder dropout");

// DISTRIBUTED TRAINING
DEFINE_bool(enable_distributed, false, "enable distributed training");
DEFINE_int64(
    world_rank,
    0,
    "rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "total number of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_string(
    rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

// FB SPECIFIC
DEFINE_string(target, "tkn", "target feature");
DEFINE_bool(everstoredb, false, "use Everstore db for reading data");
DEFINE_bool(use_memcache, false, "use Memcache for reading data");

namespace detail {

/***************************** Deprecated Flags  *****************************/
namespace {

void registerDeprecatedFlags() {
  // Register deprecated flags here using DEPRECATE_FLAGS. For example:
  // DEPRECATE_FLAGS(my_now_deprecated_flag_name, my_new_flag_name);
}

} // namespace

DeprecatedFlagsMap& getDeprecatedFlags() {
  static DeprecatedFlagsMap flagsMap = DeprecatedFlagsMap();
  return flagsMap;
}

void addDeprecatedFlag(
    const std::string& deprecatedFlagName,
    const std::string& newFlagName) {
  auto& map = getDeprecatedFlags();
  map.emplace(deprecatedFlagName, newFlagName);
}

bool isFlagSet(const std::string& name) {
  gflags::CommandLineFlagInfo flagInfo;
  if (!gflags::GetCommandLineFlagInfo(name.c_str(), &flagInfo)) {
    std::stringstream ss;
    ss << "Flag name " << name << " not found - check that it's declared.";
    throw std::invalid_argument(ss.str());
  }
  return !flagInfo.is_default;
}

} // namespace detail

void handleDeprecatedFlags() {
  auto& map = detail::getDeprecatedFlags();
  // Register flags
  static std::once_flag registerFlagsOnceFlag;
  std::call_once(registerFlagsOnceFlag, detail::registerDeprecatedFlags);

  for (auto& flagPair : map) {
    std::string deprecatedFlagValue;
    gflags::GetCommandLineOption(flagPair.first.c_str(), &deprecatedFlagValue);

    bool deprecatedFlagSet = detail::isFlagSet(flagPair.first);
    bool newFlagSet = detail::isFlagSet(flagPair.second);

    if (deprecatedFlagSet && newFlagSet) {
      // Use the new flag value
      std::cerr << "[WARNING] Both deprecated flag " << flagPair.first
                << " and new flag " << flagPair.second
                << " are set. Only the new flag will be "
                << "serialized when the model saved." << std::endl;
      ;
    } else if (deprecatedFlagSet && !newFlagSet) {
      std::cerr
          << "[WARNING] Usage of flag --" << flagPair.first
          << " is deprecated and has been replaced with "
          << "--" << flagPair.second
          << ". Setting the new flag equal to the value of the deprecated flag."
          << "The old flag will not be serialized when the model is saved."
          << std::endl;
      if (gflags::SetCommandLineOption(
              flagPair.second.c_str(), deprecatedFlagValue.c_str())
              .empty()) {
        std::stringstream ss;
        ss << "Failed to set new flag " << flagPair.second << " to value from "
           << flagPair.first << ".";
        throw std::logic_error(ss.str());
      }
    }

    // If the user set the new flag but not the deprecated flag, noop. If the
    // user set neither flag, noop.
  }
}

} // namespace w2l
