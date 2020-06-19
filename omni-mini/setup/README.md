# Setup
* This code has been tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.4.0.

## Install dependencies
### via pip
```bash
cd omni-mini/setup
pip install -r requirements.txt
```

### via conda
**Not tested yet**
```bash
conda create -n <environment_name>
conda activate <environment_name>
conda install setuptools scipy numpy Pillow h5py tqdm requests
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
pip install torchmeta
```

### via docker
```bash
cd omni-mini/setup
docker build -t <imagename> .
```
Depending on your docker setup, docker might not be able to access your host network. In that case add `--network=host`.


