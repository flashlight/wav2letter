import argparse
from collections import defaultdict
from multiprocessing import Pool

import numpy as np


TOP_K = [1]
ref_dict = {}
res_dict = defaultdict(list)


def score(x, wts):
    return (
        x["am_score"]
        + wts["tr"] * x["tr"]
        + wts["convlm"] * x["convlm"]
        + wts["len"] * x["wl_len"]
    )


def compute(wts):
    total_len = 0
    original_wer = 0.0
    oracle_wer = 0.0

    topk_wer = [0.0] * len(TOP_K)
    for sample, transcription in ref_dict.items():
        transcription_len = len(transcription)
        total_len += transcription_len
        # if sample not in res_dict:
        #    continue

        hyps = res_dict[sample]
        hyps = sorted(hyps, key=lambda x: -x["decoder_score"])
        # Original
        original_order = hyps
        original_wer += original_order[0]["wer"] * transcription_len
        # Oracle
        oracle_order = sorted(hyps, key=lambda x: x["wer"])
        oracle_wer += oracle_order[0]["wer"] * transcription_len
        # Top K
        for i, k in enumerate(TOP_K):
            order = sorted(hyps[:k], key=lambda x: -score(x, wts))
            topk_wer[i] += order[0]["wer"] * transcription_len

    return {
        "original_wer": original_wer / total_len,
        "oracle_wer": oracle_wer / total_len,
        "topk_wer": [w / total_len for w in topk_wer],
        "best_wts_trail": wts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp", type=str, required=True, help="candidates beam dump path"
    )
    parser.add_argument(
        "--list", type=str, required=True, help="data list with original transcriptions"
    )
    parser.add_argument("--in_wts", type=str, required=True, help="weights to eval")
    parser.add_argument("--convlm", type=str, default="", help="convlm ppl file path")
    parser.add_argument("--tr", type=str, default="", help="transformer ppl file path")
    parser.add_argument(
        "--search",
        action="store_true",
        help="search or not optimal weights of rescoring",
    )
    parser.add_argument(
        "--top",
        type=str,
        default="small",
        help="large beam or not, defines the topk set",
    )
    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help="use grid search instead of random search",
    )

    args = parser.parse_args()
    if args.top == "large":
        TOP_K = [2, 10, 100, 500, 1000, 2500]
    else:
        TOP_K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100, 200, 250]

    with open(args.list) as f:
        for line in f:
            data = line.strip().split()
            ref_dict[data[0]] = data[3:]

    lm_files = {
        key: open(name, "r")
        for name, key in zip([args.convlm, args.tr], ["convlm", "tr"])
        if name != ""
    }
    with open(args.hyp, "r") as f:
        for line in f:
            data = line.strip().split("|")
            # sample ID, decoder score, am score, lm score, wer
            non_transcription = data[:5]
            audio_id = non_transcription[0].strip()
            transcription = data[5].strip().split(" ")

            values_dict = {
                "wer": float(non_transcription[4]),
                "decoder_score": float(non_transcription[1]),
                "am_score": float(non_transcription[2]),
                "lm_score": float(non_transcription[3]),
                "wl_len": len(transcription) + len("".join(transcription)),
            }
            for key in ["convlm", "tr"]:
                if key in lm_files:
                    values_dict[key] = float(
                        lm_files[key].readline().strip().split(" ")[1]
                    )
                else:
                    values_dict[key] = 0.0
            res_dict[audio_id].append(values_dict)
    for f_d in lm_files.values():
        f_d.close()

    best_wts = {
        name: float(val)
        for val, name in zip(args.in_wts.split(","), ["tr", "convlm", "len"])
    }

    if args.search:
        print("searching", flush=True)
        min_wer = 100

        weights = []
        if args.gridsearch:
            # w1: tr LM weight
            # w2: convlm weight
            # w3: word score
            for w1 in [i for i in np.arange(0.0, 1.0, 0.1)]:
                for w2 in [i for i in np.arange(-0.3, 0.3, 0.1)]:
                    for w3 in [i for i in np.arange(0.0, 1.0, 0.1)]:
                        weights.append({"tr": w1, "convlm": w2, "len": w3})
        else:
            for _ in range(1000):
                weights.append(
                    {
                        "tr": np.random.rand() * 2.5,
                        "convlm": (np.random.rand() - 0.5) * 2,
                        "len": (np.random.rand() - 0.5) * 6,
                    }
                )

        num_tries = len(weights)
        print("Total number of search points", num_tries)
        threads = 50
        pool = Pool(threads)
        results = pool.map(compute, weights)
        pool.close()
        pool.join()

        assert len(results) == len(weights)

        for result in results:
            if min(result["topk_wer"]) < min_wer:
                min_wer = min(result["topk_wer"])
                best_wts = result["best_wts_trail"]

        print(best_wts, min_wer)

    best_result = compute(best_wts)

    print("| Original WER", best_result["original_wer"])
    print("| Oracle WER", best_result["oracle_wer"])
    for i, k in enumerate(TOP_K):
        print("| Top-{} rescored WER".format(k), best_result["topk_wer"][i])
