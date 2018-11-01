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
SESS=str(args.sess)
epoch=args.epoch

# normalize epoch and lr decay step
#epoch = int(np.round(3600/args.nout))*20
#args.lr_decay_step = int(np.round(3600/args.nout))*8

# check bbox data. creat bbox data if does not exist.
import os
if not os.path.isfile("output/pascal_voc_"+target+"-res18.pkl"):
    if not os.path.isfile("models/"+args.net+"/pascal_voc_"+target+"/faster_rcnn_500_40.pth"):
        command = "cp models/"+args.net+"/pascal_voc_"+target+"/faster_rcnn_500_40_625.pth models/"+args.net+"/pascal_voc_"+target+"/faster_rcnn_500_40.pth"
        subprocess.call(command, shell=True)
    command = "python demo-and-eval-save.py --coco True --writeout True --net "+args.net+" --dataset pascal_voc_"+target+" --cuda --checksession 500 --checkepoch 40 --checkpoint 625 --image_dir /data2/lost+found/img/"+target+"_val/ --truth output/baseline/"+target+"val-res101.pkl"
    subprocess.call(command, shell=True)

# mine good training data
#command = "python mining_traindata.py --dataset "+target+" --nout "+str(args.nout)
#subprocess.call(command, shell=True)

# remove cache
command = "rm data/cache/*"
subprocess.call(command, shell=True)

# replace train script
#command = "cp trainval_mine_"+target+".txt data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/trainval_"+target+".txt"
#subprocess.call(command, shell=True)

# do training
command = "python trainval_net_ds_savemod.py --cuda --dataset pascal_voc_"+target+" --net "+args.net+" --r True --s "+SESS+" --checkepoch 40 --checkpoint 625 --checksession 500 --epoch "+str(epoch)+"  --bs 1 --nw 8  --lr 1e-4 --lr_decay_step "+str(args.lr_decay_step)
subprocess.call(command, shell=True)

# evaluate
if target=="coral":
    point=1798
elif target=="taipei2":
    point=165
elif target=="jackson2":
    point=1797
elif target=="castro3":
    point=1800
elif target=="kentucky":
    point=1800
else:
    point=1
point=str(point)

end = int(np.round(int(epoch)/10))

for ep in range(1, end): 
    epoch = str(ep*10)
    command = "python demo-and-eval-save.py --nout "+str(args.nout)+" --net "+args.net+" --dataset pascal_voc_"+target+" --cuda --checksession "+SESS+" --checkepoch "+epoch+" --checkpoint "+point+" --image_dir /data2/lost+found/img/"+target+"_val/ --truth output/baseline/"+target+"val-res101.pkl"
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
