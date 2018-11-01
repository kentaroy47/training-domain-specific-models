import subprocess


targets = ["coral", "taipei2", "jackson2", "kentucky", "castro3", "dog"]
#targets=["dog"]
net = "squeeze"
session = "10"
dataset="pascal_voc_all"
epoch="30"

for target in targets:
    command = "python demo-and-eval-save.py --net "+net+" --dataset "+dataset+" --cuda --checksession "+session+" --checkepoch "+epoch+" --checkpoint 1 --image_dir /data2/lost+found/img/"+target+"_val --truth output/baseline/"+target+"val-res101.pkl"
    subprocess.call(command,shell=True)
