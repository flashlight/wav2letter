"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import random
from collections import namedtuple

import sox

Speaker = namedtuple("Speaker", ["id", "gender"])
FileRecord = namedtuple("FileRecord", ["fid", "length", "speaker"])


def split_audio(line):
    apath, meetid, hset, spk, start, end, transcript = line.strip().split(" ", 6)
    key = "_".join([meetid, hset, spk, start, end])
    os.makedirs(os.path.join(apath, "segments", meetid), exist_ok=True)
    idx = hset[-1]
    fn = f"{meetid}.Headset-{idx}.wav"
    infile = os.path.join(apath, meetid, fn)
    assert os.path.exists(infile), f"{infile} doesn't exist"
    new_path = os.path.join(apath, "segments", meetid, key + ".flac")
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(
        file_type="flac", encoding="signed-integer", bits=16, rate=16000
    )
    start = float(start)
    end = float(end)
    sox_tfm.trim(start, end)
    sox_tfm.build(infile, new_path)
    sx_dur = sox.file_info.duration(new_path)
    if sx_dur is not None and abs(sx_dur - end + start) < 0.5:
        return [meetid, key, new_path, str(round(sx_dur * 1000, 2)), transcript.lower()]


def do_split(all_records, spkrs, total_seconds, handles_chosen=None):
    """
    Greedily selecting speakers, provided we don't go over budget
    """
    time_taken = 0.0
    records_filtered = []
    idx = 0
    speakers = copy.deepcopy(spkrs)
    current_speaker_time = {spk: 0 for spk in speakers}
    current_speaker_idx = {spk: 0 for spk in speakers}
    while True:
        if len(speakers) == 0:
            break
        speaker = speakers[idx % len(speakers)]
        idx += 1
        tocontinue = False
        while True:
            cur_spk_idx = current_speaker_idx[speaker]
            if cur_spk_idx == len(all_records[speaker]):
                speakers.remove(speaker)
                tocontinue = True
                break
            cur_record = all_records[speaker][cur_spk_idx]
            current_speaker_idx[speaker] += 1
            if handles_chosen is None or cur_record.fid not in handles_chosen:
                break
        if tocontinue:
            continue
        records_filtered.append(cur_record)
        time_taken += cur_record.length
        current_speaker_time[speaker] += cur_record.length
        if abs(time_taken - total_seconds) < 10:
            break

    return records_filtered, time_taken


def get_speakers(train_file):
    cache = {}
    all_speakers = []
    with open(train_file) as f:
        for line in f:
            spl = line.split()
            speaker_id = spl[0].split("_")[2]
            gender = speaker_id[0]
            if gender not in ["M", "F"]:
                continue
            if speaker_id not in cache:
                cache[speaker_id] = 1
                speaker = Speaker(id=speaker_id, gender=gender)
                all_speakers.append(speaker)
    return all_speakers


def get_fid2length(train_file):
    fids = []
    lengths = []
    with open(train_file) as f:
        for line in f:
            spl = line.split()
            fids.append(spl[0])
            lengths.append(float(spl[2]) / 1000)
    return list(zip(fids, lengths))


def full_records(speakers, fid2length, subset_name=None):
    all_records = []
    speakers = {(speaker.id, speaker) for speaker in speakers}

    for fid, length in fid2length:
        speaker = fid.split("_")[2]
        assert speaker in speakers, f"Unknown speaker! {speaker}"

        speaker = speakers[speaker]

        if subset_name is not None:
            assert subset_name == speaker.subset
        frecord = FileRecord(speaker=speaker, length=length, fid=fid)
        all_records.append(frecord)
    return all_records


def get_speaker2time(records, lambda_key, lambda_value):
    from collections import defaultdict

    key_value = defaultdict(int)

    for record in records:
        key = lambda_key(record)
        value = lambda_value(record)
        key_value[key] += value

    return key_value


def create_limited_sup(list_dir):
    random.seed(0)
    train_file = os.path.join(list_dir, "train.lst")
    assert os.path.exists(train_file)

    speakers = get_speakers(train_file)
    print("Found speakers", len(speakers))

    write_records = {}
    chosen_records = {}

    fid2length = get_fid2length(train_file)
    all_records = full_records(speakers, fid2length)

    for gender in ["M", "F"]:
        print(f"Selecting from gender {gender}")
        records = [rec for rec in all_records if rec.speaker.gender == gender]

        speaker2time = get_speaker2time(
            records, lambda_key=lambda r: r.speaker.id, lambda_value=lambda r: r.length
        )

        # select 15 random speakers
        min_minutes_per_speaker = 15
        speakers_10hr = {
            r.speaker.id
            for r in records
            if speaker2time[r.speaker.id] >= min_minutes_per_speaker * 60
        }
        speakers_10hr = sorted(speakers_10hr)
        random.shuffle(speakers_10hr)
        speakers_10hr = speakers_10hr[:15]

        print(f"Selected speakers from gender {gender} ", speakers_10hr)

        cur_records = {}
        for speaker in speakers_10hr:
            cur_records[speaker] = [r for r in records if r.speaker.id == speaker]
            random.shuffle(cur_records[speaker])

        # 1 hr as 6 x 10min splits
        key = "10min_" + gender
        write_records[key] = {}
        for i in range(6):
            speakers_10min = random.sample(set(speakers_10hr), 3)
            write_records[key][i], _ = do_split(
                cur_records, speakers_10min, 10 * 60 / 2, chosen_records
            )
            for kk in write_records[key][i]:
                chosen_records[kk.fid] = 1

        # 9 hr
        key = "9hr_" + gender
        write_records[key], _ = do_split(
            cur_records, speakers_10hr, (9 * 60 * 60) / 2, chosen_records
        )

    train_lines = {}
    with open(train_file) as f:
        for line in f:
            train_lines[line.split()[0]] = line.strip()

    print("Writing 6 x 10min list files...")
    for i in range(6):
        with open(os.path.join(list_dir, f"train_10min_{i}.lst"), "w") as fo:
            for record in write_records["10min_M"][i] + write_records["10min_F"][i]:
                fo.write(train_lines[record.fid])

    print("Writing 9hr list file...")
    with open(os.path.join(list_dir, "train_9hr.lst"), "w") as fo:
        for record in write_records["9hr_M"] + write_records["9hr_F"]:
            fo.write(train_lines[record.fid])
