## Getting Libri-Light Book Titles

First, [download the Libri-Light dataset](https://github.com/facebookresearch/libri-light) (all three subsets `small`, `medium` and `large`).

Extract and combine the JSON metadata from each subset, grab JSON files:

```
find small medium large -type f -name "*.json" > json.metadata.manifest
```

[Install `jq`](https://stedolan.github.io/jq/).

Then generate a title list:

```
cat all.json.metadata.out | \
    xargs -I{} --max-procs=$(nproc) -n 1 \
    bash get_title_url_table.sh \
    {} \
    libri-light_unnormalized_title_to_url_text_source.json
```

Perform the following steps:

* Remove leading and trailing quotes
* Remove leading and trailing whitespace
* Remove spaces before URLs
* Remove very broken URLs

```
cat libri-light_unnormalized_title_to_url_text_source.json | rev | \
    cut -c2- | rev | cut -c2- | \
    sort -S 75% --parallel=$(nproc) | uniq | \
    awk '{$1=$1;print}' | sed "s/| h/|h/g" | grep -v -F " http" \
    > libri-light_unnormalized_title_to_url_text_source.json.clean
```

Create a title list:

```
cat libri-light_unnormalized_title_to_url_text_source.json| rev | \
    cut -c2- | rev | cut -c2- | awk -F"|" '{print $1}' | sort | uniq \
    > libri-light_all_titles.norm.uniq.sorted.lst
```

Grab Gutenberg book IDs and remove stray URL parameters and blank lines:

```
cat libri-light_unnormalized_title_to_url_text_source.json.clean | \
    grep -F "gutenberg" | sed "s#./##g" | sed "s#?.##g" | \
    sort | uniq | `grep ``-``v ``-``e ``'^$'`` ``|`` grep -Eo '[0-9]+$' | sort | uniq \`
    > libri-light_title_to_url_text_source.gutenberg_ids.final.lst
```

Resulting IDs:

```
wc -l libri-light_title_to_url_text_source.gutenberg_ids.final.lst
# 5561
```
