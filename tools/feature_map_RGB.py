from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from itertools import zip_longest
import numpy as np
import torch
import argparse
import glob
import matplotlib.pyplot as plt
import cv2
import os 
parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="config")
parser.add_argument("--model", dest="model")
args = parser.parse_args()

model = init_detector(args.config, args.model, device='cuda:0')
imgs = sorted(glob.glob('data/test/*[!_hsi][!ndvi].jpg'))

path = 'Results/' + os.path.split(args.config)[1].replace('.py','') + '/Feature_map' 

if not os.path.exists(path):
    os.makedirs(path)

for img in imgs:
    result = inference_detector(model, img)
    cfg = model.cfg
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    data = dict(img_info=dict(filename=img), img_prefix=None)
    data = test_pipeline(data)
    datas.append(data)
    data = collate(datas, samples_per_gpu=1)

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    x = model.backbone(data['img'][0].cuda())
    x = model.neck(x)

    processed = []
    for feature_map in x:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    # for fm in processed:
    #     print(fm.shape)
    
    for i in range(len(processed)):
        imgplot = plt.imshow(processed[i])
        plt.axis("off")
        plt.savefig(f"{path}/{os.path.split(img)[1].replace('.jpg','')}_{i}.jpg", bbox_inches='tight')
        # cv2.imwrite(f"{path}/{os.path.split(img)[1].replace('.jpg','')}_{i}.jpg", np.uint8(255*processed[i]))
        print(f"{path}/{os.path.split(img)[1].replace('.jpg','')}_{i}.jpg is saved!")
        # a.set_title(names[i].split('(')[0], fontsize=30)
