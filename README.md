# M<sup>2</sup>SODAI Dataset



## 1. Install Dependencies

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

## 2. Prepare Dataset



1. Download data

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vPReTPfYSLsKGUdrjqi0l_nCNDZyr5d6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vPReTPfYSLsKGUdrjqi0l_nCNDZyr5d6" -O m2sodai.zip && rm -rf /tmp/cookies.txt  
```

2. Unzip

```bash
unzip new.zip
rm new.zip
# use softlink 
ln -sf {source} {dest ex. data}
```

3. run data_tools/lableme2coco.py 

```bash
python data_tools/labelme2coco.py data/test data/test_coco --label data/label.txt
python data_tools/labelme2coco.py data/train data/train_coco --label data/label.txt
python data_tools/labelme2coco.py data/val data/val_coco --label data/label.txt
```


## 3. How to RUN?

1. Check dataset


2. Compute the mean and variance of the data set.

```bash
python ./data_tools/mean_var_calculator.py
python ./data_tools/mean_var_calculator_jpg.py
```

3. Training

```bash
python tools/train.py {config_file}
```

4. Evaluation (test)

~~~
python tools/test.py {config_file} {output_file} --eval bbox --show-score-thr 0.5
~~~


## Acknowledgment 

- This work is forked from MMdetection Repository https://github.com/open-mmlab/mmdetection
- A useful instruction of Faster R-CNN: https://www.lablab.top/post/how-does-faster-r-cnn-work-part-ii/
