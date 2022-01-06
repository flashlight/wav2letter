from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re

import sox


def process_hub5_data(sample_data):
    line, idx, hub5_sdir, hub5_audio_path, sph2pipe = sample_data
    if (not line) or line.startswith(";;") or ("IGNORE_TIME_SEGMENT_" in line):
        return None
    parts = line.strip().split()
    transcript = " ".join(parts[6:])
    transcript = transcript.replace("((", "(")
    transcript = transcript.replace("<B_ASIDE>", "")
    transcript = transcript.replace("<A_ASIDE>", "")

    spk = "{}-{}".format(parts[0], parts[1])
    start = float(parts[3])
    end = float(parts[4])
    utt = "{u}_{s}-{e}".format(
        u=spk, s="{:06d}".format(int(start * 100)), e="{:06d}".format(int(end * 100))
    )
    in_file = os.path.join(hub5_sdir, "english", parts[0] + ".sph")
    out_file = os.path.join(hub5_audio_path, "{:09d}.flac".format(idx))
    tmp_file = os.path.join(hub5_audio_path, "{pid}_tmp.wav".format(pid=os.getpid()))
    os.system(
        "{sph} -f wav -c {c} {i} {o}".format(
            sph=sph2pipe, c=1 if parts[1] == "A" else 2, i=in_file, o=tmp_file
        )
    )
    assert (
        sox.file_info.duration(tmp_file) > 0
    ), "Audio file {} duration is zero.".format(in_file)
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
    sox_tfm.trim(start, end)
    sox_tfm.build(tmp_file, out_file)
    os.remove(tmp_file)
    duration = (end - start) * 1000.0
    return "\t".join([utt, out_file, "{0:.2f}".format(duration), transcript.lower()])


def normalize_acronyms(line, acronym_dict):
    # Taken from https://git.io/fjhbu
    # Original Author - Minhua Wu

    dict_acronym = {}
    dict_acronym_noi = {}  # Mapping of acronyms without I, i
    for k, v in acronym_dict.items():
        dict_acronym[k] = v.strip()
        dict_acronym_noi[k] = v.strip()
    del dict_acronym_noi["i"]
    del dict_acronym_noi["I"]

    line = "<dummy-id> " + line.strip()
    items = line.split()
    L = len(items)
    # First pass mapping to map I as part of acronym
    for i in range(L):
        if items[i] == "i":
            x = 0
            while i - 1 - x >= 0 and re.match(r"^[A-Z]$", items[i - 1 - x]):
                x += 1

            y = 0
            while i + 1 + y < L and re.match(r"^[A-Z]$", items[i + 1 + y]):
                y += 1

            if x + y > 0:
                for bias in range(-x, y + 1):
                    items[i + bias] = dict_acronym[items[i + bias]]

    # Second pass mapping (not mapping 'i' and 'I')
    for i in range(len(items)):
        if items[i] in dict_acronym_noi.keys():
            items[i] = dict_acronym_noi[items[i]]
    return " ".join(items[1:])


def sanitize(transcript, acronym_dict):
    cleaned_words = ""
    for word in transcript.split():
        # Remove silence
        word = word.replace("[silence]", "")

        # Remove <b_aside>, <e_aside> (background conversation indicators)
        word = word.replace("<b_aside>", "")
        word = word.replace("<e_aside>", "")

        # Use special noise symbol for [vocalized-noise].
        # NOTE: Kaldi doesn't do this
        word = word.replace("[vocalized-noise]", "[noise]")

        # For word containing laughter, replace [laughter-word] by word
        # (these word are still properly understood)
        # also handle cases like [laughter-ou[r]-]
        word = re.sub(r"(-?)\[laughter\-([\S]+)\](-?)", r"\1\2\3", word)

        # for anomalous word like [Bamorghini/Lamborghini], we consider the first
        # word as it matches more with the pronounciation
        word = re.sub(r"\[(\S+)\/\S+\]", r"\1", word)
        # handle an incorrect input: 'ex[specially]-/especially]'
        word = re.sub("ex.specially...especially.", "ex-", word)

        # For partial word like -[Substi]tute use '-tute' in word transcription
        word = re.sub(r"ammu\[n\]it", r"ammu-it", word)  # handle case 'ammu[n]it[ion]-'
        word = re.sub(r"\-\[[^\]\s]+\]", r"-", word)
        word = re.sub(r"\[[^\[\s]+\]\-", r"-", word)

        # for coinages like {DJed}, {yuppyish} remove curly braces around them
        word = re.sub(r"[\{\}]+", r"", word)
        # For common alternate pronunciations like about_1 -> b aw t, them_1 eh m,
        # remove '_1'
        word = re.sub(r"_\d$", r"", word)
        word = re.sub(r"them_1's", r"them's", word)  # handle case 'them_1's'
        cleaned_words += word + " "
    # Normalize acronyms to Fisher format BBC -> b._b._c.
    return normalize_acronyms(cleaned_words, acronym_dict)


def process_swbd_data(sample_data):
    data, _, swbd_audio_path, sph2pipe, acronym_dict = sample_data
    id, sphfile, chA, chB = data
    tmp_file = os.path.join(swbd_audio_path, "{pid}_tmp.wav".format(pid=os.getpid()))
    cur_audio_path = os.path.join(swbd_audio_path, id)
    os.makedirs(cur_audio_path, exist_ok=True)
    idx = 0
    lines = []
    for channel in ["A", "B"]:
        os.system(
            "{sph} -f wav -c {c} {i} {o}".format(
                sph=sph2pipe, c=1 if channel == "A" else 2, i=sphfile, o=tmp_file
            )
        )
        assert (
            sox.file_info.duration(tmp_file) > 0
        ), "Audio file {} duration is zero.".format(sphfile)
        with open(chA if channel == "A" else chB, "r") as f:
            for line in f:
                name = line[0:6].replace("sw", "sw0")
                channel = line[6]
                splits = line.strip().split(" ", 3)
                start = float(splits[1])
                end = float(splits[2])
                transcript = sanitize(splits[3], acronym_dict)
                if not transcript:
                    continue
                utt = "{n}-{c}_{s}-{e}".format(
                    n=name,
                    c=channel,
                    s="{:06d}".format(int(start * 100 + 0.5)),
                    e="{:06d}".format(int(end * 100 + 0.5)),
                )
                out_file = os.path.join(cur_audio_path, "{:09d}.flac".format(idx))
                sox_tfm = sox.Transformer()
                sox_tfm.set_output_format(
                    file_type="flac", encoding="signed-integer", bits=16
                )
                sox_tfm.trim(start, end)
                sox_tfm.build(tmp_file, out_file)
                duration = (end - start) * 1000.0
                idx = idx + 1
                lines.append(
                    "\t".join(
                        [utt, out_file, "{0:.2f}".format(duration), transcript.lower()]
                    )
                )
        os.remove(tmp_file)
    return lines
