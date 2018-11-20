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
                      help='directory to load models', default=100)
  parser.add_argument('--sess', dest='sess',
                      help='directory to load models', default=740)
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
  parser.add_argument('--net', dest='net',
                      help='training dataset', default="res18", type=str)
  parser.add_argument('--notrain', dest='notrain',
                      help='training dataset', default=False, type=bool)

  args = parser.parse_args()
  return args

args = parse_args()
target=args.dataset
SESS=str(args.sess)

# mine good training data
#command = "python mining_traindata_classweight.py --dataset "+target+" --nout "+str(args.nout)
#subprocess.call(command, shell=True)

# remove cache
command = "rm data/cache/*"
subprocess.call(command, shell=True)

# replace train script
command = "cp trainval_"+target+".txt data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/trainval_"+target+".txt"
subprocess.call(command, shell=True)

# do training
command = "python trainval_net_ds.py --cuda --dataset pascal_voc_"+target+" --net "+args.net+" --r True --s "+SESS+" --checkepoch 40 --checkpoint 625 --checksession 500 --epoch 30  --bs 1 --nw 8  --lr 1e-4"
subprocess.call(command, shell=True)

# evaluate
point=str(1)
for ep in range(4, 13): 
    epoch = str(ep*5)
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
