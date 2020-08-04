"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

---------

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
import time

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
        print("filter out dir=", dir)
        return False
    return True

def process_experiment(dir, args):
    print("processing directory={}".format(os.path.join(args.indir, dir)))

    perfid = 1
    epoch = 1

    experiments_in_folder_stats = []
    while True:
        perf_file = os.path.join(args.indir, dir, "{0:03d}_perf".format(perfid))

        if not os.path.exists(perf_file):
            break
        with open(perf_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        if not content:
            break

        header = content[0].split()
        summary_tags = {}
        # runtimeidx = -1

        # excluded_headers = ["filter.*", "epoch", "date", "time", "epoch", "#"]
        for idx, h in enumerate(header):
        #     if not any((re.match(eh, h)) for eh in excluded_headers):
        #         if h == "runtime":
        #             h = h + " (minutes)"
        #             runtimeidx = idx - 1
            summary_tags[idx - 1] = h
        # print("summary_tags=",summary_tags)
        #summary_tags= 
        # {0: 'date', 1: 'time', 2: 'epoch', 3: 'nupdates', 4: 'lr', 5: 'lrcriterion', 6: 'runtime', 
        # 7: 'bch(ms)', 8: 'smp(ms)', 9: 'fwd(ms)', 10: 'crit-fwd(ms)', 11: 'bwd(ms)', 12: 'optim(ms)', 
        # 13: 'loss', 14: 'train-LER', 15: 'train-WER', 16: 'dev-clean-loss', 17: 'dev-clean-LER', 
        # 18: 'dev-clean-WER', 19: 'dev-other-loss', 20: 'dev-other-LER', 21: 'dev-other-WER', 22: 'avg-isz', 
        # 23: 'avg-tsz', 24: 'max-tsz', 25: 'hrs', 26: 'thrpt(sec/sec)', -1: '#'}

        # 1 / smoothing factor 
        runningAvgWeight = 9.0
        stats = {'train-WER':100.0, 'train-WER-avg':100.0, \
            'dev-other-WER':100.0,  'dev-other-WER-avg':100.0, \
                "epocs":0, "run-name":dir, "min-snr":0 , "clips":0}

        noiseArgs = re.search(kNoiseFeildsRegEx, dir)
        if noiseArgs is not None:
            stats["min-snr"] = noiseArgs.group(1)
           # stats["max_snr"] = noiseArgs.group(2)
            stats["clips"] = noiseArgs.group(3)

        expected_vals = 27
        for idx in range(1, len(content)):
            vals = content[idx].split()
            if len(vals) < max([expected_vals, train_wer_index, other_wer_index, epoc_index]):
                continue
            stats['train-WER-avg'] = (stats['train-WER-avg'] * runningAvgWeight + float(vals[train_wer_index])) \
                / (runningAvgWeight + 1.0)
            stats['train-WER'] = min(stats['train-WER'], stats['train-WER-avg'])
            stats['dev-other-WER-avg'] = (stats['dev-other-WER-avg'] * runningAvgWeight + float(vals[other_wer_index])) \
                / (runningAvgWeight + 1.0)
            stats['dev-other-WER'] = min(stats['dev-other-WER'], stats['dev-other-WER-avg'])
            stats["epocs"] = int(vals[epoc_index])
            epoch = epoch + 1
        perfid = perfid + 1

        experiments_in_folder_stats.append(stats)
    return experiments_in_folder_stats
    # stats_writer.write("{}, {}, {}\n".format(stats['dev-other-WER'], stats['train-WER'], dir))

stats_filename = ""
train_wer_index=15
other_wer_index=21
epoc_index=2
fields=["train-WER", "dev-other-WER", "epocs", "min-snr", "clips", "run-name"]

floatRegEx = "([0-9.]*)"
# Example search string:
#  "EG_GLU4x2048_S4_TR36x768_DO0.2_LD0.2_CTC_node4_noise_snr_1.2_8_clips2_saug"
# Result:
#  group(1) = min snr
#  group(2) = min snr
#  group(3) = clips
kNoiseFeildsRegEx = "snr_" + floatRegEx + "_" + floatRegEx + ".*_clips" + floatRegEx

if __name__ == "__main__":
    args = parse_and_validate_args()
    subdirs = os.listdir(args.indir)
    futures = []

    stats_filename = os.path.join(args.outdir, "stats.txt")
    if not os.path.exists(args.outdir):
        print("creating output directory={}".formate(args.outdir))
        os.makedirs(args.outdir)
    if os.path.exists(stats_filename):
        print("{}".format(stats_filename))
        os.remove(stats_filename)

    all_experiments_stats = []
    for dir_name in subdirs:
        if is_valid_dir(dir_name, args):
            try:
                experiments_in_folder_stats = process_experiment(dir_name, args)
                all_experiments_stats += experiments_in_folder_stats
            except Exception:
                print("\nError while processing the experiment in directory={}".format(dir_name))
                print(traceback.format_exc())
                sys.exit(1)

    all_experiments_stats = sorted(all_experiments_stats, key = lambda i: i['dev-other-WER']) 

    stats_writer = open(stats_filename, 'w')

    # Header
    sys.stdout.write("# ")
    stats_writer.write("# ")
    for field in fields:
        sys.stdout.write("{}, ".format(field))
        stats_writer.write("{}, ".format(field))
    sys.stdout.write("\n")
    sys.stdout.flush()
    stats_writer.write("\n")

    #  Values
    for experiment in all_experiments_stats:
        for field in fields:
            sys.stdout.write("{}, ".format(experiment[field]))
            stats_writer.write("{}, ".format(experiment[field]))
        sys.stdout.write("\n")
        sys.stdout.flush()
        stats_writer.write("\n")
    stats_writer.close()
    print("\nDone!")
