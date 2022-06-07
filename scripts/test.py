import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
from . import main_pretrain as trainer

print('success!')