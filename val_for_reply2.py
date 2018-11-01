import subprocess


#targets = ["coral", "taipei2", "jackson2", "kentucky", "castro3"]
targets=["dog"]
net = "res18"
session = "10"
dataset="pascal_voc_all"
epoch="15"

for target in targets:
    command = "python demo-and-eval-save.py --net "+net+" --dataset "+dataset+" --cuda --checksession "+session+" --checkepoch "+epoch+" --checkpoint 1 --image_dir /data2/lost+found/img/"+target+"_val --truth output/baseline/"+target+"val-res101.pkl"
    subprocess.call(command,shell=True)
