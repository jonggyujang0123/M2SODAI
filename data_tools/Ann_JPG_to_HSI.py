#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:04:49 2022

@author: jgjang
"""

import argparse
import numpy as np
import json 


parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--json',
                    help='directory for mean and var calculation')
parser.add_argument('--org_scale',
                    help='directory for mean and var calculation')
parser.add_argument('--new_scale',
                    help='directory for mean and var calculation')
args = parser.parse_args()

file_name = args.json
new_scale = float(args.new_scale)
org_scale = float(args.org_scale)

with open(file_name) as f:
    json_object = json.load(f)
    
f.close()

json_object['images']
json_object['annotations']

for i in range(len(json_object['images'])):
    json_object['images'][i]['file_name']=json_object['images'][i]['file_name'].replace('.jpg', '.mat')
    json_object['images'][i]['height'] = int(new_scale)
    json_object['images'][i]['width'] = int(new_scale)


for i in range(len(json_object['annotations'])):
    json_object['annotations'][i]['bbox'] = list(np.round(np.array(json_object['annotations'][i]['bbox']) * new_scale / org_scale))
    json_object['annotations'][i]['area'] = np.round(float(json_object['annotations'][i]['area']) * new_scale**2 / org_scale**2)
    
    
new_file_name = file_name.replace('.json', '_HSI.json')
with open(new_file_name, 'w') as json_file:
    json.dump(json_object, json_file)
    