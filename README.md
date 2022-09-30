# Project_Superresolution

A noob undergraduate trying to improve the doctor's code.

## The way to creative environment

### 1. Create a conda environment

```
conda create --name SR python=3.8
```

### 2. Get in the created environment

```
conda activate SR
```

### 3. Install Pytorch

If the system is Windows or Linux and the version of cuda is 11.6:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

If the system is Linux and the version of cuda is 11.3:

```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4. Install openCV-python

```
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 5. Install tensorboard

```
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 6. Install matplotlab

```
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 7. Install easydict

```
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 8. Install pytorch-lighting

```
pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 9. Install tifffile

```
pip install tifffile -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 10. Install skimage

```
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 11. Install torchio

```
pip install torchio -i https://pypi.tuna.tsinghua.edu.cn/simple/
```