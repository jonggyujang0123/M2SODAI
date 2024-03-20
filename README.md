<h1 align="center"> M<sup>2</sup>SODAI: Multi-Modal Ship and Floating Matter Detection Image Dataset With RGB and Hyperspectral Image Sensors
</h1>

<p align="center">
  Jonggyu Jang, Sangwoo Oh, Youjin Kim, Dongmin Seo, Youngchol Choi, Hyun Jong Yang
</p>

<p align="center">
  Conference on Neural Information Processing (NeurIPS) 2023
</p>

**Paper link:** [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8757b889350a3782b384a3ec0dfbae9-Abstract-Datasets_and_Benchmarks.html)

<img width="900" alt="image" src="https://github.com/jonggyujang0123/M2SODAI/assets/88477912/fb42288e-1662-469d-a72d-6e3ed46fc394">

---

## Updates

- 🆕 **2024/03:** We upload `the Replica of the trained weights` and `preprocessing code`. (Now, I recognize some files are wrong, I will fix it soon, until end of Mar)
- ⚠️ **2024/02: This repo requires some updates:** Now, we recognize that some files are missing when we upload the source code (data preprocessing). Until 2024/03/20, we will fix this issue by **Uploading preprocessing code, as well as **processed data** and **Replica of the trained weights**. Thanks to Another-0 and Xiaodian Zhang.

--- 

## Installation

1. Anaconda

```bash
conda create -n Maritime python=3.7
```

2. Pytorch

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install jupyterlab
jupyter server extension disable nbclassic
```

3. Other extra dependencies

```bash
pip install spectral matplotlib scikit-image fire openmim
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
# for instaboost
pip install instaboostfast
# for panoptic segmentation
pip install git+https://github.com/cocodataset/panopticapi.git
# for LVIS dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
# for albumentations
pip install -r requirements/albu.txt
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install labelme==4.5.13
```

## Prepare Dataset

- Dataset: [GDrive]() (v1.1 Will be uploaded Soon)
- Preprocessed Dataset: [Gdrive]() (will be added soon)
- Pretrained weights: [R50]() ()

Download the processed data via the above link or you can generate it yourself.

```
.M2SODAI
├── configs
├── data_tools
├── ...
├── data
│   ├── train
│   ├── test
│   ├── val
│   ├── train_coco (will be generated)
│   ├── test_coco (will be generated)
│   ├── val_coco (will be generated)
│   ├── mean_std.mat (will be generated)
│   ├── model.pkl (will be generated)
│   ├── ...
│   └── label.txt
```


- run data_tools/lableme2coco.py 

```bash
python data_tools/labelme2coco.py data/test data/test_coco --label data/label.txt
python data_tools/labelme2coco.py data/train data/train_coco --label data/label.txt
python data_tools/labelme2coco.py data/val data/val_coco --label data/label.txt
```

- make mean_std.mat

```bash
python data_tools/mean_var_calculator.py --dir data/train/
```

- make model.pkl (IPCA)

```bash
python tools/IPCA_data.py
```


## How to RUN?

- Training

```bash
python tools/train.py {config_file} 
```

ex) X50 configuration file is in `configs/faster_rcnn/faster_rcnn_x50_rgb_hsi.py`

- Evaluation

~~~
python tools/test.py {config_file} {output_file} --eval bbox --show-score-thr 0.5
~~~


## Acknowledgment 

- This work is forked from MMdetection Repository https://github.com/open-mmlab/mmdetection
- A useful instruction of Faster R-CNN: https://www.lablab.top/post/how-does-faster-r-cnn-work-part-ii/
