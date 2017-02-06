This module provides torchnet datasets that are useful when dealing with
sequential data, e.g. in language modeling or machine translation. The main
motivation for having a set of extra dataset classes is to be able to build
flexible data providers. This way, the exact data representation for training
does not need to be known at pre-processing time.  This document describes two
common data layouts for sequential data and how to implement them with the
classes provided in this module.


# Unaligned data layout
In this layout, mini-batches can contain a variable number of sequences. In the
example below, `S` marks the start of a sequence, `=` is a single element in a
sequence and `|` is a mini-batch boundary:

    S = = = = = = = | S = = = S = ...
    = = = = = S = = | = = = S = = ...
    = = = S = = = = | = = S = = = ...
    = S = = = S = = | = S = = S = ...

We're dealing with a mini-batch size of 4x8, i.e. 4 sequences are presented in
parallel, with 8 elements each. "Unaligned" refers to the fact that a sequence
can end and start at any point in the mini-batch. This layout provides high
throughput and is often used for language modeling tasks.

The main building block for this layout is `tnt.SequenceBatchDataset`. It
transforms a dataset that represents a single big sequence into multiple,
parallel sequences. In the layout above, the single big sequence is a whole
corpus of successive sequences. In `tnt.SequenceBatchDataset`, every element is
a slice across all parallel sequences. If the sequences are stored in a
`tnt.IndexedDataset`, `tnt.FlatIndexDataset` can be used to load it -- it will
represent the data as a single, continuous sequence so that it can be easily
plugged into `tnt.SequenceBatchDataset`.

The second batch dimension can be realized with a standard `tnt.BatchDataset`,
which will merge multiple successive slices into a mini-batch.


# Aligned data layout
Here, sequences only start at the beginning of a mini-batch. Thus, it may be
necessary to introduce extra padding symbols (denoted with `P` below) to obtain
two-dimensional mini-batches or to enforce a consistent mini-batch size.

    S = = = = = = = | S = = = P P P | ...
    S = = = = = P P | S = = = = = P | ...
    S = = = P P P P | S = = = = = = | ...
    S = = = = = P P | S = = = P P P | ...

This layout is useful for conditional language modeling, where the model input
is both the sequence to be modelled and a conditional input per sequence. The
total number of sequences per mini-batch is constant, so the parts of the
network that are processing the conditional input will be run on a fixed number
of inputs as well. Implementing a conditional model using an unaligned data
layout is also possible but requires additional bookkeeping and is not
necessarily more efficient.

If the underlying dataset produces separate sequences already, this layout can
be implemented using a `tnt.BatchDataset` with a custom merge function to add
the necessary padding, However, the following concepts may be helpful when
making data pipelines more efficient or flexible.

## Sorting and bucketing
It's desirable to minimize the amount of padding symbols that are presented to
the model as they will artificially inflate the training data (leading to longer
training times) without providing additional or useful information. A simple
solution is sorting the training sequences by length and constructing
mini-batches from adjacent sequences, which will be of similar length. For
conditional models, it's possible that both conditional input and the actual
sequences are of variable length (e.g. for Neural Machine Translation). In this
case, the data could be sorted according to multiple keys.

Sometimes, it can also be useful to perform less fine-grained sorting. When
constructing mini-batches from multiple sequences, it's desirable to not produce
identical mini-batches in every iteration over the dataset. This is usually done
by adding random sampling (i.e. `tnt.ShuffleDataset`) before mini-batch
assembly, but this will result in a large amount of padding for the setups
discussed here.

`tnt.BucketSortedDataset` can be used to implement all of the aforementioned
data representations. It performs a simple bucket sort with a given
`resolution`, using a user-provided `samplesize` function as the key for each
sample. Training samples will be produced by iterating over each bucket.
Optionally, samples within a bucket can also be shuffled (and re-shuffled
between iterations). By using within-bucket shuffling and varying the bucket
resolution, it's possible to compromise between minimal padding and random
presentation of samples.

An actual data provider might combine `tnt.BucketSortedDataset` with
`tnt.BatchDataset` and `tnt.ShuffleDataset` to produce random mini-batches with
minimal padding. Please note that in this case, mini-batches might contain
samples from multiple buckets; `tnt.BucketSortedDataset` merely produces a
permutation of the original dataset. If this is not desirable,
`tnt.BucketBatchDataset` can be used. It combines bucket sorting and
mini-batching and ensures that mini-batches will only contain samples from
within a single bucket.

## Truncated Backpropagation-Through-Time
When training recurrent neural networks, backpropagation-through-time (BPTT) is
used to back-propagate the gradient to each execution of network. In many
setups, it's desirable to limit the number of time-steps that the network is
executed before computing gradients and updating the parameters. Effectively,
this truncates the gradient after a fixed number of time-steps. For one,
recurrent networks often have trouble tracking long-range dependencies in the
underlying data which can prevent the error signal from being back-propagated
properly. Also, the memory cost of storing intermediate network states for very
long sequences may be prohibitive.

Implementing BPTT truncation in the early stages of a dataset provider can be
complicated. `tnt.TruncatedDatasetIterator` implements the necessary logic for
BPTT truncation and is meant to be plugged in as the final stage of a
pre-processing pipeline. For each sample which is larger than `maxsize` in the
specified `dimension`, it will split the sample along that dimension and produce
multiple partial samples instead. If the underlying dataset produces tables,
`maxsize` is only enforced on the fields specified in `fields`; other fields are
passed without modification along with each split. Table samples will also be
amended with a few flags: `_split` (sample has been produced by a split),
`_hasnext` (for all but the last part of a split) and `_cont` (for all but the
first part of a split) make it possible to control e.g. hidden state propagation
in the training code.

# Utility datasets
For sequence modeling tasks, the training objective is often to predict the next
element of a sequence. `tnt.TargetNextDataset` is a simple dataset to transform
sequence elements into `{input, target}` pairs, with `target` being a future
element.
