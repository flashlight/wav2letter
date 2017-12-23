#!/bin/sh

# fbcode only

shift # --fbcode_dir
shift # --install_dir

mkdir -p "${INSTALL_DIR}/wav2letter/utils"

cat "${FBCODE_DIR}/$1" \
    | sed -e "s/package\.searchpath('libwav2letter'\,\ package\.cpath)/'deeplearning_projects_wav2letter_libwav2letter'/g" \
    > ${INSTALL_DIR}/wav2letter/utils/init.lua
