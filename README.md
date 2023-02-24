## Structure from Motion

### Environment

- OS
    - Ubuntu 22.04.1 LTS
    - macOS 13.2

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
    - [OnePose](https://github.com/zju3dv/OnePose)

- Infra
    - [SuperPointPretrainedNetwork](https://github.com/Unbinilium/SuperPointPretrainedNetwork)
    - [SuperGluePretrainedNetwork](https://github.com/Unbinilium/SuperGluePretrainedNetwork)
    - [DIS](https://github.com/Unbinilium/DIS)
    - [COLMAP](https://github.com/Unbinilium/colmap)
    - [OpenMVS](https://github.com/Unbinilium/openMVS)
    - [VCG](https://github.com/Unbinilium/VCG)

- Extra (Not included)
    - [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils)
    - [Polyscope](https://github.com/nmwsharp/polyscope)
