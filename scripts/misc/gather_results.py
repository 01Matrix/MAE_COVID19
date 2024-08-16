import argparse
import json

from requests import head
import numpy as np
import csv
from glob import glob


# log_list = glob('./MAE_COVID19_output_finetune/2:3:5/finetune_with_mae_pretrain_vit_base/*/*.txt')
log_list = glob('./MAE_COVID19_output_finetune/2:3:5/finetune_with_mae_finetuned_vit_base/U_sani2_seed*/log.txt')

result_list=[]
for log in log_list:
    hyperpara = str(log).split('/')[-2]
    with open(log) as f:
        lines = f.readlines()
        if len(lines) == 2 or len(lines) == 1:
            continue
        last_line = lines[-1]
        mydict = eval(last_line.replace('\n',''))
        # import pdb;pdb.set_trace()
        mydict['hyperpara'] = hyperpara
        result_list.append(str(mydict))
with open('gather_results_list.txt','w') as f:
    f.writelines('\n'.join(result_list)+ '\n')

with open('./gather_results_list.csv','w',newline='') as f:
    writer=csv.writer(f,delimiter=',')
    head = ['No.','Acc','Pre','Recall','F1','AUC','Specificity','CM(TN,FP,FN,TP)','Test_loss','Total_epoch','Best_epoch','Best_val_acc','Tr_time','hyperpara']
    writer.writerow(head)
    for i in range(len(result_list)):
        if eval(result_list[i]).get('test_acc') == None:
            continue
        writer.writerow([i,eval(result_list[i])['test_acc'],eval(result_list[i])['test_pre'],eval(result_list[i])['test_recall'],\
            eval(result_list[i])['test_f1'],eval(result_list[i])['test_auc'],eval(result_list[i])['test_specificity'],\
            eval(result_list[i])['test_Confusion_Matrix(tn, fp, fn, tp)'],eval(result_list[i])['test_loss'],\
            eval(result_list[i])['Total epoch'],eval(result_list[i])['Best epoch'],eval(result_list[i])['Best val_acc'],\
            eval(result_list[i])['Training time'],eval(result_list[i])['hyperpara']])
