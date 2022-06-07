import os
from PIL import Image

img = Image.open('/mnt/sfs_turbo/jiaoxianfeng/code/ssl-pretrain/mae/datasets/large-COVID-19-ct-slice-dataset/COVID/6_Rahimzadeh_137covid_patient1_SR_4_IM00055.png')
imgSize = img.size
print(imgSize)