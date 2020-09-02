# Downloading and Preprocessing Gutenberg Books
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
