## Preprocessing Gutenberg Metadata

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
python scripts/get_titles.py \
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

Normalize the title text:
```
normalize.sh title_id_map.titles.out > title_id_map.titles.out.norm
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
