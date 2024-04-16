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

Because I rarely check this issue tab, please email me via `jgjang0123 [at] gmail [dot] com` or fill out [this form](https://forms.gle/fNu5fic3wSC16KU37).



## News

- **2024/04:** We correct this repo. Thanks to `Another-0` and `Xiaodian Zhang`. We sincerely apologize for being later than the promised time, and we truly appreciate your continued interest in our work.
- **2024/02: This repo requires some updates:** Now, we recognize that some files are missing when we upload the source code (data preprocessing). We will fix this issue by **Uploading preprocessing code**, as well as **processed data** and **Replica of the trained weights**. 

--- 

## 1. Installation

1. Anaconda

```bash
conda create -n Maritime python=3.7
```

2. Pytorch

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
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
```

## 2. Prepare Dataset

- Dataset: [GDrive](https://drive.google.com/file/d/1yGDveAVqwus_cMltHnwaR_Lx97zIatoG/view?usp=sharing)

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
│   ├── train_coco
│   ├── test_coco
│   ├── val_coco
│   ├── pca_mean_std.mat
│   ├── model.pkl
│   ├── ...
│   └── label.txt
```


- (Skip) run data_tools/lableme2coco.py 

```bash
python data_tools/labelme2coco.py data/train data/train_coco --label data/label.txt
python data_tools/labelme2coco.py data/test data/test_coco --label data/label.txt
python data_tools/labelme2coco.py data/val data/val_coco --label data/label.txt
```

- (Skip) make `pca_mean_std.mat` and IPCA. 

```bash
python tools/IPCA_data.py
```


## 3. How to RUN?

- Training

```bash
python tools/train.py {config_file} 
```

ex) R50 configuration file is in `configs/faster_rcnn/faster_rcnn_r50_rgb_hsi.py`
ex) `python tools/train.py configs/faster_rcnn_faster_rcnn_r50_rgb_hsi.py`
- Evaluation

~~~
python tools/test.py {config_file} {ckpt_file} --eval bbox
~~~


### 3.1. Replica of Trained Weights
- R-50-RGB-HSI
  - mAP: 43.7
  - Model [(link)](https://drive.google.com/file/d/1yFmdFjg-Cb3mDlsg7LcTtVq3EewN9Aq_/view?usp=sharing)
- R-50-RGB
  - mAP: 37.7
  - Model [(link)](https://drive.google.com/file/d/1yFkNq1imh_ajxcY9pq7GEE23md0gB9mC/view?usp=sharing)
- R-50-HSI (The mAP is enhanced while correcting our code)
  - mAP: 13.2
  - Model [(link)](https://drive.google.com/file/d/1yFE_yEZdQysPF1JQWBzvKRhyjg0CQHOG/view?usp=sharing)


## 4. TODO LIST

- [ ] Something... 


## Acknowledgment 

- This work is forked from MMdetection Repository https://github.com/open-mmlab/mmdetection
- A useful instruction of Faster R-CNN: https://www.lablab.top/post/how-does-faster-r-cnn-work-part-ii/
