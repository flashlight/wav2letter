# LM Corpus Reproduction

This document describes reproduction steps to create the "no overlap" language model corpus, which contains the LibriSpeech LM corpus excluding any books that are contained in LibriVox audio used in the paper.

To run some of the below, a virtual environment with the corresponding `requirements.txt` in this directory is needed. Install those dependencies with:
```
pip install -r requirements.txt
```

# Removing LibriVox Books from the LibriSpeech LM Corpus

This corpus includes even more stringent metadata checking for Gutenberg books not otherwise present in normal metadata.

First, follow the instructions for [downloading and preprocessing the Gutenberg corpus](download_preprocess_gutenberg.md).

### Finding Overlapping IDs

Generate Gutenberg IDs from the [LibriSpeech metadata](http://www.openslr.org/resources/11/librispeech-lm-corpus.tgz):

```
awk -F"|" '{print $1}' BOOKS.TXT | grep -Fv ";" | awk '{$1=$1;print}' | \
    sort | uniq > librispeech_lm_corpus.gutenberg.ids.lst
```

Follow the instructions in ([Getting Libri-Light Book Titles](./get_libri-light_book_titles.md)) to generate a list of Gutenberg IDs in the LibriSpeech LM corpus that need to be excluded:

```
comm -12 <(sort librispeech_lm_corpus.gutenberg.ids.lst) \
    <(sort librilight_title_to_url_text_source.gutenberg_ids.final.lst) \
    > librispeech_minus_librivox.metadata_only.excluded.ids.lst
```

### Creating a More Comprehensive Metadata Table

Some of the books in the LibriSpeech LM corpus aren’t always present in the metadata tables we developed previously. Creating a more comprehensive table allows us to get more potential fuzzy matches. First, generate a list of available metadata files from the cache downloaded in [Preprocessing Gutenberg Metadata](preprocessing_gutenberg_metadata.md):

```
cd cache && find . -maxdepth 1 -type d | \
    xargs --max-procs=80 -n 1 -I{} basename {} | \
    awk '$1 == ($1+0)' | \
    > ../gutenberg_metadata_dir_manifest.lst
```

Then generate title tables for all of those IDs, even if we don’t have texts:

```
python3 get_titles.py \
    --infile gutenberg_metadata_dir_manifest.lst
    --outfile /tmp/gutenberg_metadata_dir_manifest.lst.raw
```

Postprocess to remove some bad data or data without titles:

```
cat /tmp/gutenberg_metadata_dir_manifest.lst.raw | \
    awk -F"|" '{$1=$1;print}' | awk -F"|" '!length($2)' \
    > gutenberg.full_from_metadata.table
```

Normalize titles from the resulting table:

```
awk -F"|" '{print $2}' gutenberg.full_from_metadata.table \
    > gutenberg.full_from_metadata.norm.titles.lst
```

Normalize the resulting titles using the following normalization script:

```
source ../lm_corpus_and_PL_generation/normalize_title.sh gutenberg.full_from_metadata.norm.titles.lst gutenberg.full_from_metadata.norm.table
```

Grab IDs:

```
awk -F"|" '{print $1}' gutenberg.full_from_metadata.norm.table \
    > gutenberg.full_from_metadata.norm.ids.lst
```

Combine to produce the final table:

```
paste -d "|" \
    gutenberg.full_from_metadata.norm.ids.lst \
    gutenberg.full_from_metadata.norm.titles.lst.norm \
    > gutenberg.full_from_metadata.table.norm
```

### Running Fuzzy Matching

First, pull out a list of normalized Gutenberg titles for the books in the LibriSpeech LM corpus.

```
python3 join_ids.py \
    --basefile librispeech_lm_corpus.gutenberg.ids.lst \
    --tablefile  gutenberg.full_from_metadata.table.norm \
    --separator "|" \
    > librispeech_lm_corpus.gutenberg.ids.mapped_to_gutenberg_normalized_titles.table
```

Grab titles only from the resulting data:

```
awk -F"|" '{print $2}' \
    librispeech_lm_corpus.gutenberg.ids.mapped_to_gutenberg_normalized_titles.table \
    > librispeech_lm_corpus.gutenberg.ids.mapped_to_gutenberg_normalized_titles.titles.lst
```

Assuming you’ve done [Get Libri-Light Book Titles](get_libri-light_book_titles.md), create a title split set from LibriVox titles. Splitting into `NPIECES=([lines in title file] + $(nproc) - 1)/($(nproc))` is recommend for optimal parallelization:

```
mkdir librivox_title_splits && cd librivox_title_splits && \
    split --lines=$NPIECES librilight_all_titles.norm.uniq.sorted.lst
```

Run fuzzy matching:

```
find librivox_title_splits/ -type f | xargs --max-procs=80 -n 1 -I{} \
    sh -c "levenshtein-word.pl {} librispeech_lm_corpus.gutenberg.ids.mapped_to_gutenberg_normalized_titles.titles.lst > {}.librispeech_lm_corpus.minus_librivox.norm.word.distances"
```

Combine the distance scores into a single file:

```
cat *.librispeech_lm_corpus.minus_librivox.norm.word.distances \
    > combined.librispeech_lm_corpus.minus_librivox.norm.word.distances.lst
```

Filter word distances by a criteria based on their score and distance ratio:

```
python3 filter_distances.py \
    --infile combined.librispeech_lm_corpus.minus_librivox.norm.word.distances.lst \
    --score 0.3 \
    --distance_ratio 0.75
```

Manually analyze results. Results of analysis are in: https://gist.github.com/jacobkahn/998ad885eefcef2c8f27fc6d8a620fe4.

Download the raw file:

```
wget https://gist.githubusercontent.com/jacobkahn/998ad885eefcef2c8f27fc6d8a620fe4/raw/4bed6e39074e9d0350ae70044ca4d1e241ae5811/librispeech_lm_corpus.librivox.manual.excluded.titles.lst
```

Grab titles from the final result:

```
awk -F"|" '{print $2}'  \
    librispeech_lm_corpus.librivox.manual.excluded.titles.lst \
    > librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.titles
```

Run a join with the resulting titles to get corresponding Gutenberg IDs:

```
join -t "|" -2 2 \
    <(sort librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.titles) \
    title_id_map.sort_by_title.norm.table \
    > librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.table
```

Grab IDs from the table:

```
awk -F"|" '{print $2}' \
    librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.table \
    > librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.ids
```

### Combine into Final Exclusion List

Combine IDs from manual and metadata-based exclusions:

```
cat \
    librispeech_minus_librivox.metadata_only.excluded.ids.lst \
    librispeech_lm_corpus.minus_librivox.manual.excluded.gutenberg.ids \
    | sort | uniq \
    > librispeech_lm_corpus.minus_librivox.metadata_and_manual.exclude.gutenberg.ids
```

We’re excluding at most 2444 titles from the final corpus (we may not have books for many of these IDs):

```
wc -l librispeech_lm_corpus.minus_librivox.metadata_and_manual.exclude.gutenberg.ids
# 2919
```

### Excluding from the LM Corpus

```
comm -23 \
    <(sort librispeech_lm_corpus.gutenberg.ids.lst) \
    <(sort librispeech_lm_corpus.minus_librivox.metadata_and_manual.exclude.gutenberg.ids) \
    | sort | uniq \
    > librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.ids
```

The final corpus contains `12046` books, which is 83% of the original corpus:

```
wc -l librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.ids
# 11576
```

### Recover Original Missing LibriSpeech LM Corpus Books

Before recovering the final books, we need to recover books that we’re missing:

```
comm -23 \
    <(sort librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.ids) \
    <(sort title_id_map.sort_by_id.ids.norm.table) \
    > librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.missing_ids
```

Recover [most of] the books (make sure your virtual environment is activated):

```
mkdir librispeech_lm_corpus_missing_raw_utf8_noheader && \
    python3 get_gb_books_by_id.py \
    --idfile librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.missing_ids \
    --outdir librispeech_lm_corpus_missing_raw_utf8_noheader
```

One straggler:

```
cd librispeech_lm_corpus_missing_raw_utf8_noheader && \
    wget http://www.gutenberg.org/files/3103/3103-0.txt && \
    mv 3103-0.txt 3103.txt
```

Combine the corpus:

```
cat *.txt > ../librispeech_lm_corpus_missing_raw_utf8_noheader.corpus.txt
```

### Creating the Final Corpus

Combine missing books with known books. First combine known books:

```
cat librispeech_lm_corpus.minus_librivox.metadata_and_manual.gutenberg.ids \
    | xargs -I{} cat [path_to_data]/{}.body.txt \
    >> librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt.prelim
```

Combine with the missing books corpus to create the **final corpus:**

```
cat \
    librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt.prelim
    librispeech_lm_corpus_missing_raw_utf8_noheader.corpus.txt \
    > librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt
```
