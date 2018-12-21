"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def parser(filename):
    # refer to af_dtype enum here
    # http://arrayfire.org/docs/defines_8h.htm#a023d8ac325fb14f1712a52fb0940b1d5
    dtype = {
        0: 'f4',
        1: 'c8',
        2: 'f8',
        3: 'c16',
        4: 'bool',
        5: 'i4',
        6: 'u4',
        7: 'u1',
        8: 'i8',
        9: 'u8',
        10: 'i2',
        11: 'u2',
    }

    with open(filename, "rb") as binary_file:
        # refer to header definition at
        # http://arrayfire.org/docs/group__stream__func__save.htm
        binary_file.read(1)  # 1 byte for version
        array_cnt_bytes = binary_file.read(4)
        array_cnt = int.from_bytes(array_cnt_bytes, byteorder='little')

        result = []
        for _ac in range(array_cnt):
            key_length_bytes = binary_file.read(4)
            key_length = int.from_bytes(key_length_bytes, byteorder='little')
            key_bytes = binary_file.read(key_length)
            key = key_bytes.decode("utf-8")

            binary_file.read(8)  # 8 bytes for offsets

            af_dtype_byte = binary_file.read(1)
            af_dtype = int.from_bytes(af_dtype_byte, byteorder='little')

            dims = [1] * 4
            for i in range(4):
                dim_byte = binary_file.read(8)
                dim = int.from_bytes(dim_byte, byteorder='little')
                dims[i] = 1 if dim == 0 else dim

            array = np.fromfile(
                binary_file,
                dtype=np.dtype(dtype[af_dtype]),
                count=np.prod(np.array(dims)))

            array = array.reshape(dims)

            result.append((key, array))

    return result
