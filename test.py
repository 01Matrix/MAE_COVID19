import util.misc as misc
import os
print(os.environ['WORLD_SIZE'])
print(misc.get_rank()) # gobal_rank, 单节点时等于rank
print(os.environ["RANK"]) 
print(os.environ["LOCAL_RANK"]) # 只有多卡时才有