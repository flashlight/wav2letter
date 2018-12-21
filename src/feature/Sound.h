/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

namespace speech {

struct SoundInfo {
  int64_t frames;
  int64_t samplerate;
  int64_t channels;
};

// valid formats:
// wav      Microsoft WAV format (little endian)
// aiff     Apple/SGI AIFF format (big endian).
// au       Sun/NeXT AU format (big endian).
// raw      RAW PCM data.
// paf      Ensoniq PARIS file format.
// svx      Amiga IFF / SVX8 / SV16 format.
// nist     Sphere NIST format.
// voc      VOC files.
// ircam    Berkeley/IRCAM/CARL
// w64      Sonic Foundry's 64 bit RIFF/WAV
// mat4     Matlab (tm) V4.2 / GNU Octave 2.0
// mat5     Matlab (tm) V5.0 / GNU Octave 2.1
// pvf      Portable Voice Format
// xi       Fasttracker 2 Extended Instrument
// htk      HMM Tool Kit format
// sds      Midi Sample Dump Standard
// avr      Audio Visual Research
// wavex    MS WAVE with WAVEFORMATEX
// sd2      Sound Designer 2
// flac     FLAC lossless file format
// caf      Core Audio File format
// wve      Psion WVE format
// ogg      Xiph OGG container
// mpc2k    Akai MPC 2000 sampler
// rf64     RF64 WAV file

// valid subformats:
// pcm_s8      Signed 8 bit data
// pcm_16      Signed 16 bit data
// pcm_24      Signed 24 bit data
// pcm_32      Signed 32 bit data
// pcm_u8      Unsigned 8 bit data (WAV and RAW only)
// float       32 bit float data
// double      64 bit float data
// ulaw        U-Law encoded.
// alaw        A-Law encoded.
// ima_adpcm   IMA ADPCM.
// ms_adpcm    Microsoft ADPCM.
// gsm610      GSM 6.10 encoding.
// vox_adpcm   Oki Dialogic ADPCM encoding.
// g721_32     32kbs G721 ADPCM encoding.
// g723_24     24kbs G723 ADPCM encoding.
// g723_40     40kbs G723 ADPCM encoding.
// dwvw_12     12 bit Delta Width Variable Word encoding.
// dwvw_16     16 bit Delta Width Variable Word encoding.
// dwvw_24     24 bit Delta Width Variable Word encoding.
// dwvw_n      N bit Delta Width Variable Word encoding.
// dpcm_8      8 bit differential PCM (XI only)
// dpcm_16     16 bit differential PCM (XI only)
// vorbis      Xiph Vorbis encoding.

SoundInfo loadSoundInfo(const char* filename);

template <typename T>
extern std::vector<T> loadSound(const char* filename);

template <typename T>
extern void saveSound(
    const char* filename,
    std::vector<T> input,
    int64_t samplerate,
    int64_t channels,
    const char* format,
    const char* subformat);
} // namespace speech
