# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import csv
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torchaudio
from lst_utils import FileInfo, get_speakers_list, save_lst
from tqdm import tqdm


PUNCTUATION = (string.punctuation + "¡¿").replace("'", "").replace("-", "")
PUNCTUATION += "–…»“«·—’”„"


def get_size_audio_file(path_file: Path) -> float:
    r"""
    Give the size in hours on the given sequence
    """
    try:
        info = torchaudio.info(str(path_file))[0]
    except RuntimeError:
        return 0
    return info.length / (info.rate * 3600)


def to_wav2letterFormat(data: torch.tensor, sr: int) -> torch.tensor:
    r"""
    Wav2letter needs mono 16kHz inputs
    """
    if len(data.size()) == 2:
        data = data.mean(dim=0, keepdim=True)
    elif len(data.size()) == 1:
        data = data.view(1, -1)
    else:
        raise ValueError("Invalid tensor format")
    if sr != 16000:
        data = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(data)
        data = torch.clamp(data, min=-1.0, max=1.0)
    return data


def get_base_data_from_csv(pathTSV) -> List[Dict[str, str]]:
    out = []
    with open(pathTSV, "r", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, dialect="excel-tab")
        for row in reader:
            speaker_id = row["client_id"]
            name = row["path"]
            text = row["sentence"]
            out.append({"speaker_id": speaker_id, "local_path": name, "text": text})
    return out


def norm_text(
    text: str,
    char_set: Set[str],
    replace_set: Optional[Dict[str, str]] = None,
    del_set: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    text = text.lower()
    if replace_set is not None:
        for char_, val in replace_set.items():
            text = text.replace(char_, val)

    if del_set is not None:
        for char_ in del_set:
            text = text.replace(char_, "")

    valid = True
    for char_ in text.replace(" ", ""):
        if char_ not in char_set:
            valid = False
            break

    return text, valid


def load_letters(path_letter: Path):
    with open(path_letter, "r") as file:
        data = file.readlines()

    return [x.strip() for x in data]


def get_full_audio_data(
    path_dir_audio: Path,
    base_data: List[Dict[str, str]],
    char_set: Set[str],
    replace_set: Optional[Dict[str, str]] = None,
    del_set: Optional[Set[str]] = None,
    file_extension: str = None,
) -> List[FileInfo]:
    output = []
    for audio_data in tqdm(base_data, total=len(base_data)):
        path_audio = path_dir_audio / audio_data["local_path"]

        if file_extension is not None:
            path_audio = path_audio.with_suffix(file_extension)

        if not path_audio.is_file():
            continue

        size_sec = get_size_audio_file(path_audio)
        text, status = norm_text(
            audio_data["text"], char_set, replace_set=replace_set, del_set=del_set
        )
        output.append(
            FileInfo(
                size=size_sec,
                path_=path_audio,
                id_=path_audio.stem,
                text=text,
                speaker=audio_data["speaker_id"],
            )
        )

    print(f"{len(output)} files found out of {len(base_data)}")
    return output


def convert_audio_data(
    input_list: List[FileInfo], out_dir_audio: Path
) -> List[FileInfo]:
    out_dir_audio.mkdir(exist_ok=True)
    output = []
    for file_info in tqdm(input_list, total=len(input_list)):
        audio, sr = torchaudio.load(str(file_info.path_))
        audio = to_wav2letterFormat(audio, sr)

        path_out = (out_dir_audio / file_info.path_.name).with_suffix(".flac")
        torchaudio.save(str(path_out), audio, 16000)
        output.append(
            FileInfo(
                size=file_info.size,
                path_=path_out,
                id_=file_info.id_,
                text=file_info.text,
                speaker=file_info.speaker,
            )
        )

    return output


def load_filter(path_filter: Path) -> List[str]:
    with open(path_filter, "r") as f:
        return [x.strip() for x in f.readlines()]


def filter_data_by_id(input_lst: List[FileInfo], to_filter: List[str]):
    input_lst.sort(key=lambda x: x.id_)
    to_filter.sort()

    index_filter = 0
    len_filter = len(to_filter)
    out = []
    for lst_data in input_lst:
        id_ = lst_data.id_
        while index_filter < len_filter and to_filter[index_filter] < id_:
            index_filter += 1

        if index_filter >= len_filter:
            break

        if to_filter[index_filter] == id_:
            out.append(lst_data)

    print(f"{len(out)} files out of {len(to_filter)}")

    return out


def main(args):
    letters = load_letters(Path(args.path_tokens))
    data = get_base_data_from_csv(Path(args.path_tsv))
    audio_data = get_full_audio_data(
        Path(args.path_audio),
        data,
        char_set=set(letters),
        del_set=PUNCTUATION,
        file_extension=args.file_extension,
    )

    if args.path_filter is not None:
        filter_ids = load_filter(Path(args.path_filter))
        audio_data = filter_data_by_id(audio_data, filter_ids)

    if args.path_conversion is not None:
        audio_data = convert_audio_data(audio_data, Path(args.path_conversion))

    speakers = get_speakers_list(audio_data)
    print(f"{len(speakers)} speakers found")
    save_lst(audio_data, args.path_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the lst input files for common voices datasets"
    )
    parser.add_argument(
        "--path_tsv",
        type=str,
        default="/private/home/mriviere/Common_voices/en/dev.tsv",
        help="Path to the target tsv file",
    )
    parser.add_argument(
        "--path_audio",
        type=str,
        default="/private/home/mriviere/Common_voices/en/clips_16k",
        help="Path to the directory containing the audio data",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        required=True,
        help="Output lst file.",
    )
    parser.add_argument(
        "--path_tokens",
        type=str,
        default="/checkpoint/mriviere/VoxPopuli/segmentation_output/en/en_grapheme.tokens",
        help="Path to the token file",
    )
    parser.add_argument(
        "--path_filter",
        type=str,
        default=None,
        help="If given, path to a file containing the files ids to keep.",
    )
    parser.add_argument(
        "--path_conversion",
        type=str,
        default=None,
        help="If given, path to a directory where the audio should be converted",
    )
    parser.add_argument("--file_extension", type=str, default=".mp3")

    args = parser.parse_args()
    main(args)
