from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

from dataset_utils import (
    create_transcript_dict_from_listfile,
    write_transcript_list_to_file,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def pair_transcripts_with_existing_list(transcript_list, listpath):
    transcripts = create_transcript_dict_from_listfile(listpath)
    merged = {}
    for pred in transcript_list:
        merged[pred.sid] = transcripts[pred.sid]
        merged[pred.sid].transcript = pred.prediction
    # remove transcripts for which we don't have a prediction (those that were removed)
    return merged


def compute_ngrams(inp, size):
    return [inp[i : i + size] for i in range(len(inp) - (size - 1))]


def filter_transcripts(transcript_list, args):
    # fastpath
    if not args.filter:
        return transcript_list

    filtered_transcripts = []
    for transcript in transcript_list:
        good = True
        # skip transcripts with warnings
        if args.warnings:
            if transcript.warning:
                good = False
                if args.print_filtered_results:
                    eprint(
                        "Filtering predicted transcript (warning) "
                        + transcript.sid
                        + ": "
                        + transcript.prediction
                    )
                continue

        if args.ngram:
            plist = transcript.prediction.split(" ")
            # look for repeating n-grams
            ngrams = [" ".join(c) for c in compute_ngrams(plist, args.ngram_size)]
            for gram in ngrams:
                if transcript.prediction.count(gram) > args.ngram_appearance_threshold:
                    good = False
                    if args.print_filtered_results:
                        eprint(
                            "Filtering predicted transcript (ngram fail) "
                            + transcript.sid
                            + ": "
                            + transcript.prediction
                        )
                    break

        # passes all checks
        if good:
            filtered_transcripts.append(transcript)

    return filtered_transcripts


class TranscriptPrediction(object):
    def __init__(self, sid, prediction, transcript, warning=False):
        self.sid = sid
        self.prediction = prediction
        self.transcript = transcript
        self.warning = warning


def create_transcript_set(inpath, viterbi=False, distributed_decoding=False):
    with open(inpath, "r") as f:
        if not distributed_decoding:
            # first line is chronos job
            f.readline()

        predictions = []
        while True:
            # each glob has
            # - actual transcript
            # - predicted transcript
            # - actual word pieces
            # - predicted word pieces
            transcript = f.readline()
            # check if EOF
            if not transcript:
                break
            # each set is four lines, unless there's a warning
            warning = False
            if "[WARNING]" in transcript:
                transcript = f.readline()  # read an extra line to compensate
                warning = True
            transcript = transcript[
                transcript.find("|T|: ") + len("|T|: ") :
            ]  # remove |T|:
            predicted = f.readline()  # predicted transcript
            predicted = predicted[
                predicted.find("|P|: ") + len("|P|: ") :
            ]  # remove |P|:
            if viterbi:
                predicted = predicted.replace(" ", "").replace("_", " ")
                transcript = transcript.replace(" ", "").replace("_", " ")
            # if distributed_decoding:
            #     predicted = predicted[1:].replace("_", " ")

            # if not viterbi:
            # read wp
            f.readline()
            f.readline()
            sample_info = f.readline()
            if not sample_info.strip():
                continue
            sid = sample_info.split(" ")[1]
            sid = sid[:-1]
            predictions.append(
                TranscriptPrediction(sid, predicted, transcript, warning)
            )

        return predictions


def run():
    parser = argparse.ArgumentParser(
        description="Converts decoder output into train-ready list-style"
        " dataset formats"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to decoder output containing transcripts",
    )
    parser.add_argument(
        "-p",
        "--listpath",
        type=str,
        required=True,
        help="Path of existing list file dataset or which to replace transcripts",
    )
    parser.add_argument(
        "-w",
        "--warnings",
        action="store_true",
        help="Remove transcripts with EOS warnings by default",
    )
    parser.add_argument(
        "-g",
        "--ngram",
        action="store_true",
        help="Remove transcripts with ngram issues",
    )
    parser.add_argument(
        "-n",
        "--ngram_appearance_threshold",
        type=int,
        required=False,
        default=4,
        help="The number of identical n-grams that must appear in a "
        "prediction for it to be thrown out",
    )
    parser.add_argument(
        "-s",
        "--ngram_size",
        type=int,
        required=False,
        default=2,
        help="The size of n-gram which will be used when searching for duplicates",
    )
    parser.add_argument(
        "-f", "--filter", action="store_true", help="Run some filtering criteria"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output filepath"
    )
    parser.add_argument(
        "-d",
        "--distributed_decoding",
        action="store_true",
        help="Processing a combined transcript with distributed decoding",
    )
    parser.add_argument(
        "-v",
        "--print_filtered_results",
        type=bool,
        required=False,
        default=False,
        help="Print transcripts that are filtered based on filter criteria to stderr",
    )
    parser.add_argument(
        "-q",
        "--viterbi",
        action="store_true",
        help="Expects a transcript format that is consistent with a Viterbi run",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise Exception("'" + args.input + "' - input file doesn't exist")
    if not os.path.isfile(args.listpath):
        raise Exception("'" + args.input + "' - listpath file doesn't exist")

    transcripts_predictions = create_transcript_set(
        args.input, args.viterbi, args.distributed_decoding
    )
    filtered_transcripts = filter_transcripts(transcripts_predictions, args)
    final_transcript_dict = pair_transcripts_with_existing_list(
        filtered_transcripts, args.listpath
    )
    write_transcript_list_to_file(final_transcript_dict, args.output)


if __name__ == "__main__":
    run()
