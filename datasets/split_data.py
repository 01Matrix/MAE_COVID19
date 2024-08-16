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

data_dir = './data_dir/public_medical_images/selected4finetune/UCSD-AI4H-COVID-CT-data/Images-processed/U_orig'
covid_path = os.path.join(data_dir, 'COVID.txt')
non_covid_path = os.path.join(data_dir, 'non-COVID.txt')
with open(covid_path,'r',encoding = 'UTF-8') as f:
    covid_img_paths = f.readlines()

with open(non_covid_path,'r',encoding = 'UTF-8') as f:
    non_covid_img_paths = f.readlines()

covid_img_paths = [p.strip().replace('\t0', '').replace('\t1', '') for p in covid_img_paths]
non_covid_img_paths = [p.strip().replace('\t0', '').replace('\t1', '') for p in non_covid_img_paths]

covid_train_num, covid_val_num = int(0.8*len(covid_img_paths)), int(0.1*len(covid_img_paths))
covid_test_num = len(covid_img_paths) - covid_train_num - covid_val_num
covid_train_set, covid_val_set, covid_test_set = data.random_split(covid_img_paths, [covid_train_num, covid_val_num, covid_test_num])


non_covid_train_num, non_covid_val_num = int(0.8*len(non_covid_img_paths)), int(0.1*len(non_covid_img_paths))
non_covid_test_num = len(non_covid_img_paths) - non_covid_train_num - non_covid_val_num
non_covid_train_set, non_covid_val_set, non_covid_test_set = data.random_split(non_covid_img_paths, [non_covid_train_num, non_covid_val_num, non_covid_test_num])


def copy_file(path_list, target_dir):
    for path in tqdm(path_list):
        file_name = path.split('/')[-1]
        shutil.copyfile(path, os.path.join(target_dir, file_name))

