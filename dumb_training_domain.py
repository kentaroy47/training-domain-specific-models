#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:56:10 2018

@author: ken
"""

import numpy as np
import argparse
import subprocess

# simple Bbox loss
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--nout', dest='nout',
                      help='directory to load models', default=100, type=int)
  parser.add_argument('--sess', dest='sess',
                      help='directory to load models', default=730)
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
  parser.add_argument('--net', dest='net',
                      help='training dataset', default="res18", type=str)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='training dataset', default="10", type=int)
  parser.add_argument('--epoch', dest='epoch',
                      help='training dataset', default="40", type=int)

  args = parser.parse_args()
  return args

args = parse_args()
target=args.dataset
SESS=args.sess

# just acquire n training data
import glob    
import os 
trainfile = 'trainval_dumb_'+target+'.txt'
datasource = '/data2/lost+found/img/'+target+'_train/*'

#with open(datasource) as f:
files = sorted(glob.glob(datasource))

trainvals = []
outlists = []
MAX_SAMPLE = 10000
for file in files:
     trainvals.append(os.path.basename(file)[:-4])  

with open(trainfile, "w") as f: 
    for n, trainval in enumerate(trainvals):
        if n<int(args.nout):
            f.write(trainval + '\n')   

# remove cache
command = "rm data/cache/*"
subprocess.call(command, shell=True)

# replace train script
command = "cp trainval_dumb_"+target+".txt data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/trainval_"+target+".txt"
subprocess.call(command, shell=True)

# do training
command = "python trainval_net_ds_savemod.py --cuda --dataset pascal_voc_"+target+" --net "+args.net+" --r True --s "+SESS+" --checkepoch 40 --checkpoint 625 --checksession 500 --epoch 30  --bs 1 --nw 8  --lr 1e-4"
subprocess.call(command, shell=True)

# evaluate
if target=="coral":
    point=1798
elif target=="taipei2":
    point=175
elif target=="jackson2":
    point=1797
elif target=="castro3":
    point=1800
elif target=="kentucky":
    point=1800
else:
    point=1
point=str(point)

for ep in range(2, 4): 
    epoch = str(ep*10)
    command = "python demo-and-eval-save.py --net "+args.net+" --dataset pascal_voc_"+target+" --cuda --checksession "+SESS+" --checkepoch "+epoch+" --checkpoint "+point+" --image_dir /data2/lost+found/img/"+target+"_val/ --truth output/baseline/"+target+"val-res101.pkl"
    subprocess.call(command, shell=True)
#20
#25
#30
#35
#40
#45
#50
#55
#60
    
# clean up
command = "cp trainval_"+target+".txt data/VOCdevkit2007"+target+"/VOC2007/ImageSets/Main/trainval_"+target+".txt"
subprocess.call(command, shell=True)
command = "rm data/cache/*"
subprocess.call(command, shell=True)
