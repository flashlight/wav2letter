# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class FileInfo:
    id_: str
    path_: Path
    size: float
    text: str
    wer: Optional[str] = None
    ler: Optional[str] = None
    speaker: Optional[str] = None


def save_lst(lst_data: List[FileInfo], path_out: Path) -> None:

    with open(path_out, "w") as file:
        for data in lst_data:
            file.write(f"{data.id_} {data.path_} {data.size*3600 * 1000} {data.text}\n")


def load_lst(path_file: Path) -> List[FileInfo]:

    with open(path_file, "r") as file:
        data = [x.strip() for x in file.readlines()]

    out = []
    for line in data:
        tab = line.split()
        id_, path_, size = tab[:3]
        text = " ".join(tab[3:])
        out.append(FileInfo(id_, path_, float(size) / 3600 / 1000, text))

    return out


def get_speakers_list(files_data: List[FileInfo]) -> Set[str]:
    return {x.speaker for x in files_data}
