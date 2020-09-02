from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import numpy
import torch
from fairseq.data import Dictionary
from fairseq.models.fconv_lm import FConvLanguageModel
from fairseq.models.transformer_lm import TransformerLanguageModel


def load_lm(lm_path, model_type, dict_path):
    path, checkpoint = os.path.split(lm_path)
    if model_type == "convlm":
        model_handle = FConvLanguageModel.from_pretrained(
            path, checkpoint, os.path.split(dict_path)[0]
        )
    elif model_type == "transformer":
        model_handle = TransformerLanguageModel.from_pretrained(
            path, checkpoint, os.path.split(dict_path)[0]
        )
    else:
        raise Exception(
            "Unsupported language model type: use 'convlm' or 'transformer' models"
        )
    model = model_handle.models[0].decoder.cuda()
    model.eval()
    print(model)
    return model


def predict_batch(sentences, model, fairseq_dict, max_len):
    encoded_input = []
    padded_input = []
    ppls = []

    total_loss = 0.0
    nwords = 0
    for sentence in sentences:
        encoded_input.append([fairseq_dict.index(token) for token in sentence])
        assert (
            len(encoded_input[-1]) <= max_len
        ), "Error in the input length, it should be less than max_len {}".format(
            max_len
        )
        if len(encoded_input[-1]) < max_len:
            padded_input.append(
                [fairseq_dict.eos()]
                + encoded_input[-1]
                + [fairseq_dict.eos()] * (max_len - len(encoded_input[-1]))
            )
        else:
            padded_input.append([fairseq_dict.eos()] + encoded_input[-1])
    x = torch.LongTensor(padded_input).cuda()
    with torch.no_grad():
        y = model.forward(x)[0]
        if model.adaptive_softmax is not None:
            logprobs = (
                model.adaptive_softmax.get_log_prob(y, None).detach().cpu().numpy()
            )
        else:
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()

    for index, input_i in enumerate(encoded_input):
        loss = numpy.sum(logprobs[index, numpy.arange(len(input_i)), input_i])
        loss += logprobs[index, len(input_i), fairseq_dict.eos()]
        ppls.append(loss)

        total_loss += loss
        nwords += len(input_i) + 1
    return ppls, total_loss, nwords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running forward pass for language model"
    )
    parser.add_argument("--model", required=True, type=str, help="path to the model")
    parser.add_argument(
        "--dict", required=True, type=str, help="path to the dict of the model"
    )
    parser.add_argument(
        "--max-tokens",
        required=True,
        type=int,
        default=1024,
        help="max tokens in the batch",
    )
    parser.add_argument(
        "--text", required=True, type=str, help="path to text to be evaluated"
    )
    parser.add_argument(
        "--out", type=str, default="out.txt", help="path to text to be saved"
    )
    parser.add_argument(
        "--skip",
        type=bool,
        default=False,
        help="skip <sampleID> <decoder score> <AM score> tokens",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        type=str,
        help="Language model type, supported values 'convlm' and 'transformer'",
    )
    args = parser.parse_args()

    fairseq_dict = Dictionary.load(args.dict)
    model = load_lm(args.model, args.model_type, args.dict)

    total_loss = 0.0
    nwords = 0.0
    batch = []
    original_lines = []
    max_len = 0
    with open(args.text, "r") as ftext, open(args.out, "w") as fout:
        for line in ftext:
            # id | decoder score | am score | lm score | wer | transcription
            line_parsed = line.rstrip().split("|")
            sentence = line_parsed[-1].strip().split(" ")
            if (len(batch) + 1) * numpy.maximum(
                max_len, len(sentence)
            ) > args.max_tokens:
                if len(batch) == 0:
                    if args.skip:
                        original_lines.append(line_parsed[0].strip().split(" ")[0])
                    batch.append(sentence)
                    max_len = len(sentence)
                    continue
                ppls, loss_batch, nwords_batch = predict_batch(
                    batch, model, fairseq_dict, max_len
                )
                total_loss += loss_batch
                nwords += nwords_batch
                for index in range(len(batch)):
                    if args.skip:
                        fout.write(original_lines[index] + " {}\n".format(ppls[index]))
                    else:
                        fout.write("{}\n".format(ppls[index]))
                batch = [sentence]
                if args.skip:
                    original_lines = [line_parsed[0].strip().split(" ")[0]]
                max_len = len(sentence)
            else:
                batch.append(sentence)
                if args.skip:
                    original_lines.append(line_parsed[0].strip().split(" ")[0])
                max_len = numpy.maximum(max_len, len(sentence))
        if len(batch) > 0:
            ppls, loss_batch, nwords_batch = predict_batch(
                batch, model, fairseq_dict, max_len
            )
            total_loss += loss_batch
            nwords += nwords_batch
            for index in range(len(batch)):
                if args.skip:
                    fout.write(original_lines[index] + " {}\n".format(ppls[index]))
                else:
                    fout.write("{}\n".format(ppls[index]))

    print("Total PPL", numpy.exp(-total_loss / nwords))
