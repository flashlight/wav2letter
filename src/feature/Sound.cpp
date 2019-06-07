/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Sound.h"

#include <string>
#include <unordered_map>

#include <sndfile.h>

namespace {
const std::unordered_map<std::string, int> formats{
    {"wav", SF_FORMAT_WAV},     {"aiff", SF_FORMAT_AIFF},
    {"au", SF_FORMAT_AU},       {"raw", SF_FORMAT_RAW},
    {"paf", SF_FORMAT_PAF},     {"svx", SF_FORMAT_SVX},
    {"nist", SF_FORMAT_NIST},   {"voc", SF_FORMAT_VOC},
    {"ircam", SF_FORMAT_IRCAM}, {"w64", SF_FORMAT_W64},
    {"mat4", SF_FORMAT_MAT4},   {"mat5", SF_FORMAT_MAT5},
    {"pvf", SF_FORMAT_PVF},     {"xi", SF_FORMAT_XI},
    {"htk", SF_FORMAT_HTK},     {"sds", SF_FORMAT_SDS},
    {"avr", SF_FORMAT_AVR},     {"wavex", SF_FORMAT_WAVEX},
    {"sd2", SF_FORMAT_SD2},     {"flac", SF_FORMAT_FLAC},
    {"caf", SF_FORMAT_CAF},     {"wve", SF_FORMAT_WVE},
    {"ogg", SF_FORMAT_OGG},     {"mpc2k", SF_FORMAT_MPC2K},
    {"rf64", SF_FORMAT_RF64}};

const std::unordered_map<std::string, int> subformats{
    {"pcm_s8", SF_FORMAT_PCM_S8},       {"pcm_16", SF_FORMAT_PCM_16},
    {"pcm_24", SF_FORMAT_PCM_24},       {"pcm_32", SF_FORMAT_PCM_32},
    {"pcm_u8", SF_FORMAT_PCM_U8},       {"float", SF_FORMAT_FLOAT},
    {"double", SF_FORMAT_DOUBLE},       {"ulaw", SF_FORMAT_ULAW},
    {"alaw", SF_FORMAT_ALAW},           {"ima_adpcm", SF_FORMAT_IMA_ADPCM},
    {"ms_adpcm", SF_FORMAT_MS_ADPCM},   {"gsm610", SF_FORMAT_GSM610},
    {"vox_adpcm", SF_FORMAT_VOX_ADPCM}, {"g721_32", SF_FORMAT_G721_32},
    {"g723_24", SF_FORMAT_G723_24},     {"g723_40", SF_FORMAT_G723_40},
    {"dwvw_12", SF_FORMAT_DWVW_12},     {"dwvw_16", SF_FORMAT_DWVW_16},
    {"dwvw_24", SF_FORMAT_DWVW_24},     {"dwvw_n", SF_FORMAT_DWVW_N},
    {"dpcm_8", SF_FORMAT_DPCM_8},       {"dpcm_16", SF_FORMAT_DPCM_16},
    {"vorbis", SF_FORMAT_VORBIS}};
} // namespace

namespace speech {

SoundInfo loadSoundInfo(const char* filename) {
  SNDFILE* file;
  SF_INFO info;

  /* mandatory */
  info.format = 0;

  if (!filename) {
    throw std::invalid_argument("loadSoundInfo: filename is null");
  }

  if (!(file = sf_open(filename, SFM_READ, &info))) {
    throw std::runtime_error(
        "loadSoundInfo: unknown format or could not open file - " +
        std::string(filename));
  }

  sf_close(file);

  SoundInfo usrinfo;
  usrinfo.frames = info.frames;
  usrinfo.samplerate = info.samplerate;
  usrinfo.channels = info.channels;
  return usrinfo;
}

template <typename T>
extern std::vector<T> loadSound(const char* filename) {
  SNDFILE* file;
  SF_INFO info;

  info.format = 0;

  if (!filename) {
    throw std::invalid_argument("loadSound: filename is null");
  }

  if (!(file = sf_open(filename, SFM_READ, &info))) {
    throw std::runtime_error(
        "loadSound: unknown format or could not open file - " +
        std::string(filename));
  }

  std::vector<T> in(info.frames * info.channels);
  sf_count_t nframe;
  if (std::is_same<T, float>::value) {
    nframe =
        sf_readf_float(file, reinterpret_cast<float*>(in.data()), info.frames);
  } else if (std::is_same<T, double>::value) {
    nframe = sf_readf_double(
        file, reinterpret_cast<double*>(in.data()), info.frames);
  } else if (std::is_same<T, int>::value) {
    nframe = sf_readf_int(file, reinterpret_cast<int*>(in.data()), info.frames);
  } else if (std::is_same<T, short>::value) {
    nframe =
        sf_readf_short(file, reinterpret_cast<short*>(in.data()), info.frames);
  } else {
    throw std::logic_error("loadSound: called with unsupported T");
  }
  sf_close(file);
  if (nframe != info.frames) {
    throw std::runtime_error(
        "loadSound: read error - " + std::string(filename));
  }
  return in;
}

template <typename T>
extern void saveSound(
    const char* filename,
    std::vector<T> input,
    int64_t samplerate,
    int64_t channels,
    const char* format,
    const char* subformat) {
  SNDFILE* file;
  SF_INFO info;

  if (formats.find(format) == formats.end()) {
    throw std::invalid_argument(
        "saveSound: invalid format - " + std::string(format));
  }
  if (subformats.find(subformat) == subformats.end()) {
    throw std::invalid_argument(
        "saveSound: invalid subformat - " + std::string(subformat));
  }

  info.channels = channels;
  info.samplerate = samplerate;
  info.format =
      formats.find(format)->second | subformats.find(subformat)->second;

  if (!filename) {
    throw std::invalid_argument("saveSound: filename is null");
  }

  if (!(file = sf_open(filename, SFM_WRITE, &info))) {
    throw std::runtime_error(
        "saveSound: invalid format or could not write file - " +
        std::string(filename));
  }

  sf_count_t nframe;

  if (std::is_same<T, float>::value) {
    nframe = sf_writef_float(
        file, reinterpret_cast<float*>(input.data()), info.frames);
  } else if (std::is_same<T, double>::value) {
    nframe = sf_writef_double(
        file, reinterpret_cast<double*>(input.data()), info.frames);
  } else if (std::is_same<T, int>::value) {
    nframe =
        sf_writef_int(file, reinterpret_cast<int*>(input.data()), info.frames);
  } else if (std::is_same<T, short>::value) {
    nframe = sf_writef_short(
        file, reinterpret_cast<short*>(input.data()), info.frames);
  } else {
    throw std::logic_error("saveSound: called with unsupported T");
  }
  sf_close(file);
  if (nframe != info.frames) {
    throw std::runtime_error(
        "saveSound: write error - " + std::string(filename));
  }
}
} // namespace speech

template std::vector<float> speech::loadSound(const char*);
template std::vector<double> speech::loadSound(const char*);
template std::vector<int> speech::loadSound(const char*);
template std::vector<short> speech::loadSound(const char*);

template void speech::saveSound(
    const char*,
    std::vector<float>,
    int64_t,
    int64_t,
    const char*,
    const char*);
template void speech::saveSound(
    const char*,
    std::vector<double>,
    int64_t,
    int64_t,
    const char*,
    const char*);
template void speech::saveSound(
    const char*,
    std::vector<int>,
    int64_t,
    int64_t,
    const char*,
    const char*);
template void speech::saveSound(
    const char*,
    std::vector<short>,
    int64_t,
    int64_t,
    const char*,
    const char*);
