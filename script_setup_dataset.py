
dataset="jackson"
datadir="/data2/lost+found/img/"
videodir="/data2/lost+found/video/streamlink/"
gpu=1

import glob
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
args = parser.parse_args()

dataset=args.dataset

# mkdir for train and val
traindir=datadir+dataset+"_train/"
valdir=datadir+dataset+"_val/"
command="mkdir "+traindir
subprocess.call(command, shell=True)
command="mkdir "+valdir
subprocess.call(command, shell=True)

# generate jpg by ffmpeg
if not os.path.isfile(videodir+dataset+".mp4"):
    subprocess.call("ffmpeg -i "+videodir+dataset+".ts -c:v copy -c:a aac -strict -2  "+videodir+dataset+".mp4", shell=True)
#if dataset=="jackson2":
#    if not os.path.isfile(videodir+dataset+"_crop.mp4"):
#        subprocess.call("ffmpeg -i "+videodir+dataset+".mp4 -vf crop=800:450:400:300 -strict -2 "+videodir+"jackson2_crop.mp4", shell=True)
#        subprocess.call("cp "+videodir+"jackson2_crop.mp4 "+videodir+"jackson2.mp4", shell=True)    

file=traindir+dataset+"_train0001.jpg"
if not os.path.isfile(file):
    command="ffmpeg -t 01:00:00 -i "+videodir+dataset+".mp4 -r 1.0 "+traindir+dataset+"_train%4d.jpg"
    subprocess.call(command, shell=True)
file=valdir+dataset+"_val0001.jpg"
if not os.path.isfile(file):
    command="ffmpeg -t 02:00:00 -i "+videodir+dataset+".mp4 -r 1.0 -ss 01:00:00 "+valdir+dataset+"_val%4d.jpg"
    subprocess.call(command, shell=True)

# make train and val pickle
subprocess.call("export CUDA_VISIBLE_DEVICES="+str(gpu), shell=True)
if not os.path.isfile("/home/ken/distil/output/baseline/"+dataset+"train-res101.pkl"):
    args="--coco True --net res101 --cuda --checkepoch 10 --checkpoint 9771 --image_dir "+traindir+" --outname "+dataset+"train"
    command="python demo_coco.py "+args
    subprocess.call(command, shell=True)
if not os.path.isfile("/home/ken/distil/output/baseline/"+dataset+"val-res101.pkl"):
    args="--coco True --net res101 --cuda --checkepoch 10 --checkpoint 9771 --image_dir "+valdir+" --outname "+dataset+"val"
    command="python demo_coco.py "+args
    subprocess.call(command, shell=True)

# get val results
#directory='