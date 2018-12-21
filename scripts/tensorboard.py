"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Script to process the w2l logs to tensorboard summary format and store them in
a new directory.

Usage: tensorboard.py [-h] -i INDIR [-o OUTDIR] [-f FILTER] [-a APPEND]
            [-n NTHREADS]

Arguments:
  -i INDIR, --indir INDIR
                        Input directory containing experiments
  -o OUTDIR, --outdir OUTDIR
                        Output directory to log tensorboard summaries
  -f FILTER, --filter FILTER
                        Optional regex filter for filtering experiments by
                        experiment directory name Ex: `.*timit.*`
  -a APPEND, --append APPEND
                        Optional comma-separated list of flags to be append to
                        the directory name. Ex: `lr,train`
                        Flag can be of the form `train='<some (regex)>'` too

  -n NTHREADS, --nthreads NTHREADS
                        Number threads to be used by ThreadPool
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import os
import re
import string
import sys
import traceback
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

import tensorflow as tf


def parse_and_validate_args():
    parser = argparse.ArgumentParser(description="Prepare tensorboard")
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        required=True,
        help="Input directory containing experiments",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="/home/" + os.getenv("USER") + "/w2ltboard/",
        help="Output directory to log tensorboard summaries",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Optional regex filter for filtering experiments by directory name "
        " Ex: `.*timit.*`",
    )
    parser.add_argument(
        "-a",
        "--append",
        type=str,
        default=None,
        help="Optional list of flags to be append to the directory name "
        "(for filtering experiments in Tensorboard UI) Ex: `lr,train`. "
        "Flag can be of the form `train='<some (regex)>'` in which case "
        "the first match value will be used in place of the flag value",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        type=int,
        default=10,
        help="Number threads to be used by ThreadPool",
    )

    args = parser.parse_args()

    if not os.path.exists(args.indir):
        print('Input directory"' + args.indir + "\" doesn't exist")
        exit()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return args


def is_valid_dir(dir, args):
    if not os.path.isdir(os.path.join(args.indir, dir)):
        return False
    if not os.path.exists(os.path.join(args.indir, dir, "001_perf")):
        return False
    if not glob.glob(os.path.join(args.indir, dir, "001_config*")):
        return False
    if args.filter is not None and re.match(args.filter, dir) is None:
        return False
    return True


def may_be_append_dir_name(dir, args):
    if args.append is None:
        return dir
    flags = args.append.strip().split(",")
    config_file = glob.glob(os.path.join(args.indir, dir, "001_config*"))[0]
    valid_chars = "=-_.()%s%s" % (string.ascii_letters, string.digits)
    with open(config_file) as file:
        config = file.read()
        for f in flags:
            m = re.match("(.*)='(.*)'", f)
            fre = None
            if m:
                f = m.group(1)
                fre = m.group(2)
            flg = re.search("(?<=\W)" + f + "[ ]*=.+?(?=(\\\\n--|,\n))", config)
            if flg is not None:
                lbl = flg.group(0)
                if fre:
                    lbl = re.search(fre, lbl)
                    if lbl and len(lbl.groups()) > 0:
                        lbl = f + "=" + lbl.group(1)
                    else:
                        lbl = f + "=UNK"
                dir += ";" + "".join(c for c in lbl if c in valid_chars)
    return dir


def process_experiment(dir, args):
    sys.stdout.write(".")
    sys.stdout.flush()
    if not is_valid_dir(dir, args):
        return
    basename = may_be_append_dir_name(os.path.basename(dir), args)
    writer = tf.summary.FileWriter(
        os.path.join(args.outdir, basename), graph=tf.get_default_graph()
    )
    perfid = 1
    epoch = 1
    while True:
        perf_file = os.path.join(args.indir, dir, "{0:03d}_perf".format(perfid))
        if not os.path.exists(perf_file):
            break

        with open(perf_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        header = content[0].split()
        summary_tags = {}
        runtimeidx = -1
        excluded_headers = ["filter.*", "epoch", "date", "time", "epoch", "#"]
        for idx, h in enumerate(header):
            if not any((re.match(eh, h)) for eh in excluded_headers):
                if h == "runtime":
                    h = h + " (minutes)"
                    runtimeidx = idx - 1
                summary_tags[idx - 1] = h
        for idx in range(1, len(content)):
            vals = content[idx].split()
            if runtimeidx > 0:
                hrs, min, sec = vals[runtimeidx].split(":")
                mins = int(hrs) * 60 + int(min) + float(sec) / 60
                vals[runtimeidx] = mins

            summary = tf.Summary(
                value=[
                    tf.Summary.Value(tag=v, simple_value=float(vals[k]))
                    for k, v in summary_tags.items()
                ]
            )
            writer.add_summary(summary, epoch)
            epoch = epoch + 1
        perfid = perfid + 1
    writer.flush()


if __name__ == "__main__":
    print("Tensorflow version:", tf.__version__)
    args = parse_and_validate_args()
    subdirs = os.listdir(args.indir)
    futures = []
    with ThreadPoolExecutor(max_workers=args.nthreads) as pool:
        for s in subdirs:
            futures.append(pool.submit(process_experiment, s, args))
    wait(futures, return_when=ALL_COMPLETED)
    # check for exceptions
    for i in range(len(subdirs)):
        try:
            futures[i].result()
        except Exception as e:
            print("\nError while processing the experiment :", subdirs[i])
            print(traceback.format_exc())
            sys.exit(1)
    print("\nDone!")
