#!/bin/sh

# fbcode only

shift # --fbcode_dir
shift # --install_dir

mkdir -p "${INSTALL_DIR}/beamer"

cat "${FBCODE_DIR}/$1" \
    | sed -e "s/package\.searchpath('libbeamer'\,\ package\.cpath)/'deeplearning_projects_wav2letter_libbeamer'/g" \
    > ${INSTALL_DIR}/beamer/env.lua
