/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <vector>

namespace w2l {

enum class SoundFormat {
  WAV, // Microsoft WAV format (little endian)
  AIFF, // Apple/SGI AIFF format (big endian).
  AU, // Sun/NeXT AU format (big endian).
  RAW, // RAW PCM data.
  PAF, // Ensoniq PARIS file format.
  SVX, // Amiga IFF / SVX8 / SV16 format.
  NIST, // Sphere NIST format.
  VOC, // VOC files.
  IRCAM, // Berkeley/IRCAM/CARL
  W64, // Sonic Foundry's 64 bit RIFF/WAV
  MAT4, // Matlab (tm) V4.2 / GNU Octave 2.0
  MAT5, // Matlab (tm) V5.0 / GNU Octave 2.1
  PVF, // Portable Voice Format
  XI, // Fasttracker 2 Extended Instrument
  HTK, // HMM Tool Kit format
  SDS, // Midi Sample Dump Standard
  AVR, // Audio Visual Research
  WAVEX, // MS WAVE with WAVEFORMATEX
  SD2, // Sound Designer 2
  FLAC, // FLAC lossless file format
  CAF, // Core Audio File format
  WVE, // Psion WVE format
  OGG, // Xiph OGG container
  MPC2K, // Akai MPC 2000 sampler
  RF64, // RF64 WAV file
};

enum class SoundSubFormat {
  PCM_S8, // Signed 8 bit data
  PCM_16, // Signed 16 bit data
  PCM_24, // Signed 24 bit data
  PCM_32, // Signed 32 bit data
  PCM_U8, // Unsigned 8 bit data (WAV and RAW only)
  FLOAT, // 32 bit float data
  DOUBLE, // 64 bit float data
  ULAW, // U-Law encoded.
  ALAW, // A-Law encoded.
  IMA_ADPCM, // IMA ADPCM.
  MS_ADPCM, // Microsoft ADPCM.
  GSM610, // GSM 6.10 encoding.
  VOX_ADPCM, // Oki Dialogic ADPCM encoding.
  G721_32, // 32kbs G721 ADPCM encoding.
  G723_24, // 24kbs G723 ADPCM encoding.
  G723_40, // 40kbs G723 ADPCM encoding.
  DWVW_12, // 12 bit Delta Width Variable Word encoding.
  DWVW_16, // 16 bit Delta Width Variable Word encoding.
  DWVW_24, // 24 bit Delta Width Variable Word encoding.
  DWVW_N, // N bit Delta Width Variable Word encoding.
  DPCM_8, // 8 bit differential PCM (XI only)
  DPCM_16, // 16 bit differential PCM (XI only)
  VORBIS // Xiph Vorbis encoding.
};

struct SoundInfo {
  int64_t frames;
  int64_t samplerate;
  int64_t channels;
};

SoundInfo loadSoundInfo(std::istream& f);
SoundInfo loadSoundInfo(const std::string& filename);

template <typename T>
std::vector<T> loadSound(std::istream& f);
template <typename T>
std::vector<T> loadSound(const std::string& filename);

template <typename T>
void saveSound(
    std::ostream& f,
    const std::vector<T>& input,
    int64_t samplerate,
    int64_t channels,
    const SoundFormat format,
    const SoundSubFormat subformat);

template <typename T>
void saveSound(
    const std::string& filename,
    const std::vector<T>& input,
    int64_t samplerate,
    int64_t channels,
    const SoundFormat format,
    const SoundSubFormat subformat);

} // namespace w2l
