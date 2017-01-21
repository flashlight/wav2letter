# Cleaning WSJ

In the following replace `[...]` with appropriate paths.

First, get a fresh version of [sph2pipe](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools) and compile it.

Then
```
create.lua [...]/WSJ0/media [...]/WSJ1/media -dst /tmp/wsj-idx -sph2pipe [...]/sph2pipe_v2.5/sph2pipe
```

# Creating LM

bigram:
```
convert-arpa.lua [...]/WSJ1/media/13_32.1/wsj1/doc/lng_modl/base_lm/bcb20onp.z -preprocess [...]/wav2letter/data/wsj/preprocess.lua lm-2g.arpa dict-2g.lst -r 3 -letters [...]/letters.lst
```

trigram:
```
convert-arpa.lua [...]/WSJ1/media/13_32.1/wsj1/doc/lng_modl/base_lm/tcb20onp.z -preprocess [...]/wav2letter/data/wsj/preprocess.lua lm-3g.arpa dict-3g.lst -r 3 -letters [...]/letters.lst
```
