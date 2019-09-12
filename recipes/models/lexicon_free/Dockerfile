# ==================================================================
# module list
# ------------------------------------------------------------------
# flashlight       master       (git, CUDA backend)
# python kenlm     e47088d      (git, CUDA backend)
# fairseq          22e535e      (git)
# ==================================================================

FROM wav2letter/wav2letter:cuda-base-26c69be

# ==================================================================
# flashlight https://github.com/facebookresearch/flashlight.git
# ------------------------------------------------------------------
RUN cd /root && git clone --recursive https://github.com/facebookresearch/flashlight.git && \
    cd /root/flashlight && git checkout da99018f393c9301c9bb50908dabde954b290256 && \
    git submodule update --init --recursive && mkdir -p build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA && \
    make -j8 && make install && \
# ==================================================================
# kenlm rebuild with max order 20 and install python wrapper
# ------------------------------------------------------------------
    cd /root/kenlm/build && \
    cmake .. -DKENLM_MAX_ORDER=20 && make -j8 && make install && \
    cd /root/kenlm && \
    sed -i 's/DKENLM_MAX_ORDER=6/DKENLM_MAX_ORDER=20/g' setup.py && \
    pip install . && \
# ==================================================================
# wav2letter with GPU backend
# ------------------------------------------------------------------
    cd /root && git clone --recursive https://github.com/facebookresearch/wav2letter.git && \
    export MKLROOT=/opt/intel/mkl && export KENLM_ROOT_DIR=/root/kenlm && \
    cd /root/wav2letter && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DW2L_LIBRARIES_USE_CUDA=ON -DKENLM_MAX_ORDER=20 && \
    make -j8 && \
# ==================================================================
# fairseq
# ------------------------------------------------------------------
    cd /root && git clone https://github.com/pytorch/fairseq && \
    cd fairseq && git checkout 22e535e23b4c2e95e7e72198ae98c7e0d4ac8c18 && \
    git submodule update --init --recursive && \
    pip install --editable . && \
# ==================================================================
# sph2pipe
# ------------------------------------------------------------------
    cd /root && wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz && \
    tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5 && \
    gcc -o sph2pipe *.c -lm
