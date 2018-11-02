# -*- coding: utf-8 -*-
import argparse

# args.
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--nout', dest='nout',
                      help='directory to load models', default=100)
  parser.add_argument('--sess', dest='sess',
                      help='directory to load models', default=1)
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

import subprocess
import os

# make labels with res101
print("generating labels using res101")
subprocess.call("mkdir output", shell=True)
subprocess.call("mkdir output/baseline", shell=True)

if not os.path.isfile("output/baseline/"+target+"train-res101.pkl"):
    args="--coco True --net res101 --cuda --image_dir images/"+target+"_train --outname "+target+"train"
    command="python demo_res101_coco.py "+args
    subprocess.call(command, shell=True)
else:
    print("train labels exist")
    
if not os.path.isfile("output/baseline/"+target+"val-res101.pkl"):
    args="--coco True --net res101 --cuda --image_dir images/"+target+"_val --outname "+target+"val"
    command="python demo_res101_coco.py "+args
    subprocess.call(command, shell=True)
else:
    print("val labels exist")


# prepare dataset in pascal_voc format
print("making pascal-like dataset for training")
command="python script_makevoc.py --dataset "+target
subprocess.call(command, shell=True)