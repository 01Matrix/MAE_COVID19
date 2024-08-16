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

covid_txt_path = './data_dir/public_medical_images/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_orig/COVID.txt'
non_covid_txt_path = './data_dir/public_medical_images/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_orig/non-COVID.txt'

fh = open(covid_txt_path, 'r')  #读取文件
for line in fh:
    line = line.rstrip()  #这一行就是图像的路径，以及标签
    covid_img_paths.append(line)

fh = open(non_covid_txt_path, 'r')  #读取文件
for line in fh:
    line = line.rstrip()  #这一行就是图像的路径，以及标签
    non_covid_img_paths.append(line)



train_ratio, val_ratio = 0.8, 0.1
covid_train_num, covid_val_num = int(train_ratio*len(covid_img_paths)), int(val_ratio*len(covid_img_paths))
covid_test_num = len(covid_img_paths) - covid_train_num - covid_val_num
covid_train_set, covid_val_set, covid_test_set = data.random_split(covid_img_paths, [covid_train_num, covid_val_num, covid_test_num])


non_covid_train_num, non_covid_val_num = int(train_ratio*len(non_covid_img_paths)), int(train_ratio*len(non_covid_img_paths))
non_covid_test_num = len(non_covid_img_paths) - non_covid_train_num - non_covid_val_num
non_covid_train_set, non_covid_val_set, non_covid_test_set = data.random_split(non_covid_img_paths, [non_covid_train_num, non_covid_val_num, non_covid_test_num])

def create_data_txt(path_list, target_path):
    with open(target_path, 'w+') as f:
        for ph in tqdm(path_list):
            f.writelines(ph+'\n')
    

create_data_txt(covid_train_set + non_covid_train_set, 'train.txt')
create_data_txt(covid_val_set + non_covid_val_set, 'val.txt')
create_data_txt(covid_test_set + non_covid_test_set, 'test.txt')
