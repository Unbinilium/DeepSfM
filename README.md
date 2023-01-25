## Structure from Motion

### Env

- Conda with CUDA

```bash
conda env create -n SfM -f env/conda_cuda_3.8.yml
```

- Venv with CPU

```bash
python3 -m venv . && source bin/activate
python3 -m pip install -r env/venv_cpu_3.10.txt
```
