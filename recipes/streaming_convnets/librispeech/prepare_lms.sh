#!/bin/sh
KENLM="$1"
DATA_DST="$2"

# download
wget -c -O - http://www.openslr.org/resources/11/3-gram.arpa.gz | gunzip -c > "$DATA_DST/3-gram.arpa"
wget -c -O - http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz | gunzip -c > "$DATA_DST/3-gram.pruned.1e-7.arpa"
wget -c -O - http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz | gunzip -c > "$DATA_DST/3-gram.pruned.3e-7.arpa"

# convert lo lower case
cat "$DATA_DST/3-gram.arpa" | tr '[:upper:]' '[:lower:]' > "$DATA_DST/3-gram.arpa.lower"
cat "$DATA_DST/3-gram.pruned.1e-7.arpa" | tr '[:upper:]' '[:lower:]' > "$DATA_DST/3-gram.pruned.1e-7.arpa.lower"
cat "$DATA_DST/3-gram.pruned.3e-7.arpa" | tr '[:upper:]' '[:lower:]' > "$DATA_DST/3-gram.pruned.3e-7.arpa.lower"

# convert to bin
"$KENLM/build_binary" trie "$DATA_DST/4-gram.arpa.lower" "$DATA_DST/4-gram.bin"
"$KENLM/build_binary" trie "$DATA_DST/3-gram.arpa.lower" "$DATA_DST/3-gram.bin"
"$KENLM/build_binary" trie "$DATA_DST/3-gram.pruned.1e-7.arpa.lower" "$DATA_DST/3-gram.pruned.1e-7.bin"
"$KENLM/build_binary" trie "$DATA_DST/3-gram.pruned.3e-7.arpa.lower" "$DATA_DST/3-gram.pruned.3e-7.bin"

# quantize model
"$KENLM/build_binary" trie -a 22 -q 8 -b 8 "$DATA_DST/4-gram.arpa.lower" "$DATA_DST/4-gram.bin.qt"
"$KENLM/build_binary" trie -a 22 -q 8 -b 8 "$DATA_DST/3-gram.arpa.lower" "$DATA_DST/3-gram.bin.qt"
"$KENLM/build_binary" trie -a 22 -q 8 -b 8 "$DATA_DST/3-gram.pruned.1e-7.arpa.lower" "$DATA_DST/3-gram.pruned.1e-7.bin.qt"
"$KENLM/build_binary" trie -a 22 -q 8 -b 8 "$DATA_DST/3-gram.pruned.3e-7.arpa.lower" "$DATA_DST/3-gram.pruned.3e-7.bin.qt"
