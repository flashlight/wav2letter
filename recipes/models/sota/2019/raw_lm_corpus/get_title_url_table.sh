#!/bin/bash
jq '.book_meta.title + "|" + .book_meta.url_text_source' $1 >> $2
