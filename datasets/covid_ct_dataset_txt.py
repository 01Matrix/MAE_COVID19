from PIL import Image
from torch.utils.data import Dataset
 

class COVID_CT_Dataset_txt(Dataset):
    def __init__(self, txt_path, transform = None):
        fh = open(txt_path, 'r')
        imgs = []  #用来存储路径与标签
        for line in fh:
            line = line.rstrip()  #这一行就是图像的路径，以及标签  
            
            words = line.split('\t')
            imgs.append((words[0], int(words[1])))  #路径和标签添加到列表中
            self.imgs = imgs                        
            self.transform = transform
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = Image.open(fn).convert('RGB')  #把图像转成RGB
        if self.transform is not None:
            img = self.transform(img) 
        return img, label              #这就返回一个样本
    
    def __len__(self):
        return len(self.imgs)          #返回长度，index就会自动的指导读取多少