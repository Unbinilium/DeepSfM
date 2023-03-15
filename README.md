## Deep Learning Optimized Multi-View 3D Reconstruction (SfM)

### Abstract

In order to improve the robustness, accuracy, and completeness of incremental motion recovery-based multi-view 3D reconstruction, my research studies the structure and principle of the traditional multi-view 3D reconstruction pipeline based on motion recovery and combines the cutting-edge research achievements of deep learning in the field of image processing to design, proposed a deep learning optimized multi-view 3D reconstruction pipeline.


### Environment

- OS
    - Ubuntu 22.04.1 LTS
    - macOS 13.2.1

- HW
    - NVIDIA 30 Series
    - Apple M Series

### Setup

- Repository

```bash
git clone --recurse-submodules https://github.com/Unbinilium/SfM.git
cd SfM
```

- Ubuntu Auto (with COLMAP, OpenMVS, Meshlab)

```
bash scripts/setup_ubuntu.sh
```

- Conda with CUDA

```bash
conda env create -n SfM -f env/conda_cuda_3.8.yml
```

- Venv with CPU

```bash
python3 -m venv . && source bin/activate
python3 -m pip install -r env/venv_cpu_3.10.txt
```

### Acknowledgments

- SfM Pipeline
    - [OnePose](https://github.com/zju3dv/OnePose) (heavily borrowed)

- Modules
    - [DIS](https://github.com/Unbinilium/DIS)
    - [SuperPointPretrainedNetwork](https://github.com/Unbinilium/SuperPointPretrainedNetwork)
    - [SuperGluePretrainedNetwork](https://github.com/Unbinilium/SuperGluePretrainedNetwork)
    - [COLMAP](https://github.com/Unbinilium/colmap)
    - [OpenMVS](https://github.com/Unbinilium/openMVS)
    - [VCG](https://github.com/Unbinilium/VCG)

- Extra (Not included)
    - [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils)
    - [Polyscope](https://github.com/nmwsharp/polyscope)
