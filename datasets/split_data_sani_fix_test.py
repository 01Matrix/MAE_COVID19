from tqdm import tqdm 
import os
import shutil
import glob
import pickle as pkl
import torch.utils.data as data
import torch

torch.manual_seed(42)

covid_img_paths = []
non_covid_img_paths = []

covid_path = '/mnt/sfs_turbo/public_medical_images/datasets/selected4finetune/UCSD-AI4H-COVID-CT-data/Images-processed/U_sani/COVID'
non_covid_path = '/mnt/sfs_turbo/public_medical_images/datasets/selected4finetune/UCSD-AI4H-COVID-CT-data/Images-processed/U_sani/non-COVID'


covid_img_paths = glob.glob(os.path.join(covid_path, '*.*'))
non_covid_img_paths = glob.glob(os.path.join(non_covid_path, '*.*'))

test_txt_path = '/mnt/sfs_turbo/hxiao/codes/finetune-covid19/test_list%50.txt'

covid_test_set = []
non_covid_test_set = []

fh = open(test_txt_path, 'r')  #读取文件
for line in fh:
    line = line.rstrip()  #这一行就是图像的路径，以及标签
    words = line.split('\t')
    if words[1] == '0':
        covid_img_paths.remove(words[0])
        covid_test_set.append(words[0])
    elif words[1] == '1':
        non_covid_img_paths.remove(words[0])
        non_covid_test_set.append(words[0])
    else:
        print('label error')

train_ratio = 0.2
train_ratio *= 2

covid_train_num = int(train_ratio*len(covid_img_paths))
covid_val_num = len(covid_img_paths) - covid_train_num
covid_train_set, covid_val_set= data.random_split(covid_img_paths, [covid_train_num, covid_val_num])


non_covid_train_num = int(train_ratio*len(non_covid_img_paths))
non_covid_val_num = len(non_covid_img_paths) - non_covid_train_num
non_covid_train_set, non_covid_val_set = data.random_split(non_covid_img_paths, [non_covid_train_num, non_covid_val_num])

def copy_file(path_list, target_dir):
    if not os.path.exists(target_dir):
        os,os.makedirs(target_dir)
    for path in tqdm(path_list):
        file_name = path.split('/')[-1]
        shutil.copyfile(path, os.path.join(target_dir, file_name))

copy_file(covid_train_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/train/COVID')
copy_file(non_covid_train_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/train/non_COVID')
copy_file(covid_val_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/val/COVID')
copy_file(non_covid_val_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/val/non_COVID')
copy_file(covid_test_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/test/COVID')
copy_file(non_covid_test_set, '/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/U_sani_235_fix/test/non_COVID')
