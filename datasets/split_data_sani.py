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

covid_path = './data_dir/public_medical_images/selected4finetune/UCSD-AI4H-COVID-CT-data/Images-processed/U_sani/COVID'
non_covid_path = './data_dir/public_medical_images/selected4finetune/UCSD-AI4H-COVID-CT-data/Images-processed/U_sani/non-COVID'

covid_img_paths = glob.glob(os.path.join(covid_path, '*.*'))
non_covid_img_paths = glob.glob(os.path.join(non_covid_path, '*.*'))


train_ratio, val_ratio = 0.2, 0.3

covid_train_num, covid_val_num = int(train_ratio*len(covid_img_paths)), int(val_ratio*len(covid_img_paths))
covid_test_num = len(covid_img_paths) - covid_train_num - covid_val_num
covid_train_set, covid_val_set, covid_test_set = data.random_split(covid_img_paths, [covid_train_num, covid_val_num, covid_test_num])


non_covid_train_num, non_covid_val_num = int(train_ratio*len(non_covid_img_paths)), int(train_ratio*len(non_covid_img_paths))
non_covid_test_num = len(non_covid_img_paths) - non_covid_train_num - non_covid_val_num
non_covid_train_set, non_covid_val_set, non_covid_test_set = data.random_split(non_covid_img_paths, [non_covid_train_num, non_covid_val_num, non_covid_test_num])

def copy_file(path_list, target_dir):
    if not os.path.exists(target_dir):
        os,os.makedirs(target_dir)
    for path in tqdm(path_list):
        file_name = path.split('/')[-1]
        shutil.copyfile(path, os.path.join(target_dir, file_name))


