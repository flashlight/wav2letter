# LM Corpus Reproduction

This document describes the reproduction steps to create the "no overlap" language model corpus, which contains the LibriSpeech LM corpus excluding any books that are contained in LibriVox audio used in the paper.

## Downloading the Data

Pretending you’re a search indexer and hitting the `robots` endpoint will allow you to not get rate limited by Cloudflare. `harvest` is perfect for getting everything else.

```
wget -m -H -nd "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
```

The above will take 2-3 hours to run depending on your connection. The result will be `~11 GB`, but this may change over time as books are added/removed.

### Removing and Unzipping

Remove some extra garbage:

```
rm harvest*
rm robots.txt
```

Remove duplicate encodings of books:

```
ls | grep "\-0.zip" | xargs rm
ls | grep "\-8.zip" | xargs rm
```

Remove some other weird nonsense (which have non-standard formatting):

```
rm 89-Contents.zip
rm 89-AnnexI.zip
rm 10681-body.zip
rm 5192-tex.zip
rm 10681-index.zip
rm 15824-h.zip
rm 3290-u.zip
rm 13526-page-inames.zip
rm 18251-mac.zip
rm 89-Descriptions.zip
```

A super broken book:

```
rm -rf 12hgp10a.*
```

Unzip:

```
unzip "*.zip"
rm "*.zip"
```

## Preparing Raw Files

Move text files out from nested directories (some have this):

```
mv */*.txt ./
```

A straggler:

```
mv ./10376/10376.TXT ./10376.txt
```

### Removing Cruft

Unzipping also adds media/random other stuff. Delete it, along with empty directories (or anything that’s not a `txt` file):

```
ls | grep -v "\.txt" | xargs rm -rf
```

As of December 17, 2019, the resulting data is `14 GB` and has `37445` texts.

### Install the Needed Dependencies:

```
pip install -r requirements.txt
```

## Removing Headers

The following creates a `[book_id].body.txt` given a Gutenberg `book.txt` file

```
python3 process_raw_text.py \
    --indir [dir containing txt files]
```

Combine into a new directory:

```
mkdir [some new dir]
cp *.body.txt [some new dir]
```

A few books which have extremely broken headers may not be included (this number is < 50).

## Preprocessing Metadata

### Extracting Metadata

Download `rdf` metadata files:

```
wget http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2
```

Unzip:

```
tar xjf rdf-files.tar.bz2
```

### Generating an ID List

In the directory containing the Gutenberg `txt` files with headers already stripped:

```
cd [directory] && find . -type f | sed "s/\.body.txt//g" > manifest.out
```

### Getting Titles and Creating a Title ↔ ID Map

Run the XML parsing script which uses some hard-coded XQuery/XPath to obtain title the title given the book ID.

```
python get_titles.py \
    --infile manifest.out \
    --outfile title_id_map.out
```

which produces a file that looks like:

```
10000|the magna carta
10001|apocolocyntosis
10002|the house on the borderland
...
```

### Normalizing the Title Table

Generate a sorted list of IDs and a sorted title list with the the same relative ordering:

```
awk -F"|" '{print $1}' title_id_map.out > title_id_map.ids.out
awk -F"|" '{print $1}' title_id_map.out > title_id_map.titles.out
```

Normalize the title text (see `../lm_corpus_and_PL_generation/normalize_title.sh`):
```
../lm_corpus_and_PL_generation/normalize_title.sh title_id_map.titles.out title_id_map.titles.out.norm
```

Recombine into a ID/title table:

```
paste -d "|" title_id_map.ids.out title_id_map.titles.out.norm | sort \
    > title_id_map.sort_by_id.norm.table
```

Create a map sorted by tables:

```
cat title_id_map.sort_by_id.norm.table | sort -k 2 -t "|" \
    > title_id_map.sort_by_title.norm.table
```

Create an ID list:

```
awk -F"|" '{print $1}' title_id_map.sort_by_id.norm.table \
    > title_id_map.sort_by_id.ids.norm.table
```

### Create a Split Title List

For later use in distributed fuzzy matching, split the title list up into pieces. Splitting into `NPIECES=([lines in title file] + $(nproc) - 1)/($(nproc))` is recommend for optimal parallelization:

```
mkdir gutenberg_title_splits && cd gutenberg_title_splits && \
    split --lines=$NPIECES title_id_map.titles.out.norm
```

## Generating a LibriSpeech LM Corpus with Raw Gutenberg Text

Download the LibriSpeech LM corpus from http://www.openslr.org/11/ and unzip. Extract a collection of Gutenberg IDs:

```
cut -d '|' -f 1  BOOKS.TXT | sed 's/^ *//;s/ *$//' \
    | sed '/^;/ d' > librispeech.corpus.ids
```

Repeat the below steps with this ID list.

## Removing LibriVox Books from the LibriSpeech LM Corpus


### Finding Overlapping IDs

Generate Gutenberg IDs from the LibriSpeech metadata:

```
awk -F"|" '{print $1}' BOOKS.TXT | grep -Fv ";" | awk '{$1=$1;print}' | \
    sort | uniq > librispeech_lm_corpus.gutenberg.ids.lst
```

#### Grab LibriVox Titles

First, [download the Libri-Light dataset](https://github.com/facebookresearch/libri-light) (all three subsets `small`, `medium` and `large`).

Extract and combine the JSON metadata from each subset, grab JSON files:

```
find small medium large -type f -name "*.json" > json.metadata.manifest
```

[Install `jq`](https://stedolan.github.io/jq/). Then generate a title list:

```
cat all.json.metadata.out | \
    xargs -I{} --max-procs=$(nproc) -n 1 \
    bash ./get_title_url_table.sh \
    {} \
    libri-light_unnormalized_title_to_url_text_source.json
```

Perform the following steps:

* Remove leading and trailing quotes
* Remove leading and trailing whitespace
* Remove spaces before URLs
* Remove very broken URLs

```
cat librilight_unnormalized_title_to_url_text_source.json | rev | \
    cut -c2- | rev | cut -c2- | \
    sort -S 75% —parallel=80 | uniq | \
    awk '{$1=$1;print}' | sed "s/| h/|h/g" | grep -v -F " http" \
    > light_unnormalized_title_to_url_text_source.json.clean
```

Create a title list:

```
cat librilight_unnormalized_title_to_url_text_source.json| rev | \
    cut -c2- | rev | cut -c2- | awk -F"|" '{print $1}' | sort | uniq \
    > librilight_all_titles.norm.uniq.sorted.lst
```

Grab Gutenberg book IDs and remove stray URL parameters and blank lines:

```
cat light_unnormalized_title_to_url_text_source.json.clean | \
    sed "s#./##g" | sed "s#?.##g" | sort | uniq | `grep -v -e '^$' `\
    > light_title_to_url_text_source.gutenberg_ids.final.lst
```

There are 4403 total IDs.

```
wc -l light_title_to_url_text_source.gutenberg_ids.final.lst
# 4403
```

Remove overlap:
```
comm -12 <(sort librispeech_lm_corpus.gutenberg.ids.lst) \
    <(sort librilight_title_to_url_text_source.gutenberg_ids.final.lst) \
    > librispeech_minus_librivox.metadata_only.excluded.ids.lst
```

### Creating a More Comprehensive Metadata Table

Some of the books in the LibriSpeech LM corpus aren’t always present in the metadata tables we developed previously. Creating a more comprehensive table allows us to get more potential fuzzy matches. Using the cache downloaded in the `Processing Metadata` section above, run the following:

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

Normalize the resulting titles using the following normalization script in `../lm_corpus_and_PL_generation/normalize_title.sh`.

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

Using the result of the `Get Librivox Book Titles` step above, create a title split set from LibriVox titles. Splitting into `NPIECES=([lines in title file] + $(nproc) - 1)/($(nproc))` is recommend for optimal parallelization:

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
python3 FilterDistances.py \
    --infile combined.librispeech_lm_corpus.minus_librivox.norm.word.distances.lst \
    --score 0.3 \
    --distance_ratio 0.75
```

Manually analyze results. Results of analysis are in `librispeech_lm_corpus.librivox.manual.excluded.titles.lst`.

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

Recover [most of] the books:

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
