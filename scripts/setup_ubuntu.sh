#!/bin/bash

## Create tmp dir
mkdir tmp
cd tmp
WS="$(pwd)"


## Update sources
sudo apt-get update


## Python deps
cd "${WS}"

pip3 install -r ../env/venv_cpu_3.10


## Compile COLMAP
cd "${WS}"

sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libopencv-dev

git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES='native'
ninja
sudo ninja install


## Compile OpenMVS
cd "${WS}"

#Get deps
sudo apt-get install -y libboost-iostreams-dev libglfw3-dev libopencv-dev
git clone --recurse-submodules https://github.com/cdcseacave/VCG.git

#Clone OpenMVS
git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git

#Make build directory:
cd openMVS
mkdir make
cd make

#Run CMake:
cmake .. -DVCG_ROOT="${WS}/VCG" -DOpenMVS_USE_CUDA=OFF

#Build:
cmake --build . -j4

#Install OpenMVS library (optional):
sudo cmake --install .

echo "PATH=/usr/local/bin/OpenMVS:\$PATH" >> ~/.profile
source ~/.profile


## Install meshlab
cd "${WS}"

sudo apt-get install -y meshlab
