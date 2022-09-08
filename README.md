# train_ko_model_with_tsdae

TSDAE를 통한 한국어 언어모델 훈련

### install

```sh
conda env create -n tsdae -f environment.yaml
```

### WSL에 CUDA 설치

NVIDIA 공식 가이드(<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>)에 따라 WSL cuda 설치

이후
```sh
export CUDA_HOME=/usr/local/cuda
```

`CUDA_HOME`을 설정해줌

### WSL [bitandbytes](https://github.com/TimDettmers/bitsandbytes)

bitandbytes는 리눅스에서만 사용가능

```sh
pip install bitsandbytes
```

WSL에서는

```sh
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

`LD_LIBRARY_PATH`를 추가적으로 설정해야 함
