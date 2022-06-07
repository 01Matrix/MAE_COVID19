import argparse
import torch
from torchvision import datasets,transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
parser.add_argument('--dataset', type=str, nargs='+', help='finetune dataset list')

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        print(X.shape)
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def load_pretrain(args,transform):
    dataset_name = {'C_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVIDX_CT_2A/C_orig',
                    'C_sani':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVIDX_CT_2A/C_sani',
                    'C_sani2':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVIDX_CT_2A/C_sani2',
                    'L_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/large_COVID_19_ct_slice_dataset/curated_data/L_orig',
                    'L_sani':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/large_COVID_19_ct_slice_dataset/curated_data/L_sani',
                    'L_sani2':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/large_COVID_19_ct_slice_dataset/curated_data/L_sani2',
                    'U_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_orig',
                    'U_sani':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_sani',
                    'U_sani2':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/UCSD_AI4H_COVID_CT_data/Images-processed/U_sani2',
                    'CC_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVID_19_and_common_pneumonia_chest_CT_dataset/CC_orig',
                    'CC_sani':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVID_19_and_common_pneumonia_chest_CT_dataset/CC_sani',
                    'CC_sani2':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVID_19_and_common_pneumonia_chest_CT_dataset/CC_sani2',
                    'SI_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/siim_covid19_detection_xray/SI_orig',
                    'SI_sani':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/siim_covid19_detection_xray/SI_sani',
                    'C_SI_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/C_SI_orig',
                    'S_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/sarscov2_ctscan_dataset',
                    'C1000':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4pretrain/COVID-CT/COVID19-CT-Dataset1000+',
                    'CD_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4pretrain/COVID-nonCT/COVID_DA_Xray',
                    'CQ_orig':'/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4pretrain/COVID-nonCT/COVID_QU_Ex_Dataset_Xray'
               }
    transform_dict = {
        'train': transform,
        }
    # args.dataset_path = {'data_path': dataset_name[args.dataset]}
    train_list = []
    for dataset in args.dataset:
        print(args.dataset)
        txt_path = glob('{}/*.txt'.format(dataset_name[dataset]))
        for txt in txt_path:
            with open(txt) as f:
                lines = f.readlines()
                for line in lines:
                    train_list.append(line)
    np.random.shuffle(train_list)
    trainset = CovidCTDataset(train_list,transform=transform_dict['train'])
    return trainset

class CovidCTDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - COVID
                - img1.png
                - img2.png
                - ......
            - non-COVID
                - img1.png
                - img2.png
                - ......
        """
        self.transform = transform
        self.img_list = [item.split('\t') for item in data_list]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = int(self.img_list[idx][1])
        label = torch.tensor(int(label))
        return image, label

if __name__ == '__main__':
    # train_dataset = datasets.ImageFolder(root='/sharefs/baaihealth/public_datasets/public_medical_images/datasets/selected4finetune/COVIDX_CT_2A/C_sani', transform=transforms.ToTensor())
    args = parser.parse_args()
    train_dataset = load_pretrain(args,transform=transforms.ToTensor())
    print(getStat(train_dataset))

    '''
    C_orig: ([0.46127674, 0.46127674, 0.46127674], [0.33709714, 0.33709714, 0.33709714])
    C_sani: ([0.47210848, 0.47210848, 0.47210848], [0.33132237, 0.33132237, 0.33132237])
    C_sani2: ([0.4742368, 0.4742368, 0.4742368], [0.33663702, 0.33663702, 0.33663702])

    L_orig:([0.27432978, 0.2743217, 0.2743129], [0.20456745, 0.2045658, 0.20456246])
    L_sani:([0.34898245, 0.34887907, 0.34879124], [0.22624372, 0.22623318, 0.22620067])
    L_sani:([0.35837775, 0.35836878, 0.35837382], [0.22819997, 0.22819625, 0.22815883])

    U_orig:([0.59566385, 0.59527427, 0.59510213], [0.29826742, 0.29836667, 0.29833865])
    U_snai:([0.6028448, 0.6024151, 0.6022638], [0.30355135, 0.30359438, 0.30356687])
    U_sani2:([0.59332675, 0.5928408, 0.59264743], [0.3024677, 0.30249834, 0.30248058])

    CC_orig:([0.5111813, 0.51118076, 0.51118034], [0.30600742, 0.30600557, 0.30600384])
    CC_sani:([0.4862498, 0.4862498, 0.4862498], [0.29962894, 0.29962894, 0.29962894])
    CC_sani2:([0.4812714, 0.4812714, 0.4812714], [0.29716596, 0.29716596, 0.29716596])
    
    C_orig+CC_orig: ([0.48062998, 0.48062977, 0.48062965], [0.3250353, 0.3250346, 0.32503396])
    '''