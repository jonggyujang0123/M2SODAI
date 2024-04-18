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

- [FAQ](https://github.com/jonggyujang0123/M2SODAI?tab=readme-ov-file#5-faq)
- [TODO LIST](https://github.com/jonggyujang0123/M2SODAI?tab=readme-ov-file#4-todo-list).
- [Dataset Inquire Form](https://forms.gle/oDEcL3ULFebzmy8u7)

## News

- **2024/04: Add Image Registration History.** Some people are interested in our image registration procedure (`rgb` and `hsi` data). Now, the overall procedure is available [here](https://github.com/jonggyujang0123/M2SODAI/blob/master/History_Registration/2_1_Auto_Imgreg_v2.ipynb). It is very **time-consuming** and **labor-intensive job**(I worked all night for about 4-5 days doing only this registration procedure from scratch.). So, If you mail me before starting, I can give you **tip/advice** I learned doing this.
   - Due to the policy of the research institute, we cannot provide **raw** data of RGB/HSI sensors. If you have plan for building another dataset using image registration, I can help you for free.  
- **2024/04: Correct this repo.** We correct this repo. Thanks to `Another-0` and `Xiaodian Zhang`. We sincerely apologize for being later than the promised time, and we truly appreciate your continued interest in our work. The previous code is wrong; hence, please use the current dataset/code. 
- **2024/02: Recognize mistakes in our repo.** Now, we recognize that some files are missing/wrong when we upload the source code. We will fix this issue by **Uploading preprocessing code**, as well as **processed data** and **Replica of the trained weights**. 

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

**Download**

To encourage related research, we will provide datasets upon your request. 
Please email your name and affiliation to the person in charge (`jgjang0123 [at] gmail [dot] com`). We ask for your information only to ensure that the dataset is used for **non-commercial purposes**. 
I will give you access authority within 24 hours. If not, please remind me again.

- Keyword **[BANANA]** in the subject line
- Name
- Affiliation
- Google ID to share this dataset

Do not provide this to third parties or post it publicly anywhere.

<!--
- Dataset: [GDrive](https://drive.google.com/file/d/1yGDveAVqwus_cMltHnwaR_Lx97zIatoG/view?usp=sharing)
-->

Download the processed data via the above link. 

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

- [ ] Split images by 400x400 (make 11GB GPU work).
  - One challenge is that most of the target objects are cut out doing this. Is there anyone who has an idea to solve this? 


## 5. FAQ


## Acknowledgment 

- This work is forked from MMdetection Repository https://github.com/open-mmlab/mmdetection
- A useful instruction of Faster R-CNN: https://www.lablab.top/post/how-does-faster-r-cnn-work-part-ii/

## Cite this work 

```bib
@inproceedings{NEURIPS2023_a8757b88,
	author = {Jang, Jonggyu and Oh, Sangwoo and Kim, Youjin and Seo, Dongmin and Choi, Youngchol and Yang, Hyun Jong},
	booktitle = {Advances in Neural Information Processing Systems},
	editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
	pages = {53831--53843},
	publisher = {Curran Associates, Inc.},
	title = {M\^{}\lbrace 2\rbrace SODAI: Multi-Modal Maritime Object Detection Dataset With RGB and Hyperspectral Image Sensors},
	url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/a8757b889350a3782b384a3ec0dfbae9-Paper-Datasets_and_Benchmarks.pdf},
	volume = {36},
	year = {2023},
```
