import os
import wandb
import pandas as pd
api = wandb.Api()
# 通过这些方式获取相应的值：run.id,run.name,run.tags,run.config,run.config.keys(),run.state,run.sweep,run.sweep.id
# run = api.run("bluedynamic/MAE_COVID19/run_id") 获取指定的某个run
# sweep_runs = api.sweep("bluedynamic/MAE_COVID19/sweep_id") 获取一个sweep中的所有run
# runs = api.runs("bluedynamic/MAE_COVID19") 获取整个project中的所有run

# wandb.init(project="MAE_COVID19_2", entity="bluedynamic")
# run_id = wandb.run.id
# print(run_id)

# # 或者
# run = wandb.init(project="MAE_COVID19_2", entity="bluedynamic")
# run_id = run.id

#----------
runs = api.runs("bluedynamic/MAE_COVID19")
print(len(runs))
for run in runs:
    # print(type(run.tags))
    print(run.name)
    # # print(type(run.config['block_list']))
    if run.config['finetune'] == '':
        print('changing...')
        run.tags = run.tags + ['TFS']
        run.update()
        

    # if run.config['block_list'] == "" or run.config['block_list'] == None:
        # print('changing...')
        # run.config['finetune'] = os.path.basename(run.config['finetune'])
        # run.update()

#----------
# runs = api.runs("bluedynamic/MAE_COVID19")
# summary_list, config_list, name_list = [], [], []
# for run in runs: 
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict)
#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#          if not k.startswith('_')})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })
# runs_df.to_csv("project.csv")

#----------
# sweep = api.sweep("bluedynamic/MAE_COVID19/762yqejn")
# sweep_runs = sweep.runs
# for run in sweep_runs:
#     if run.config['finetune'] == '':
#         print(type(run.config['fft']))
#         run.config['block_list'] = ''
#         run.config['fft'] = ''
#         run.config['attn'] = ''
#         run.config['mlp'] = ''
#         run.config['Tag'] = 'TFS'
#         run.update()

    # if run.config['fft'] == 'Yes' and run.config['block_list'] == '10,11':
    #     print(run)
    #     run.config['block_list'] = ''
    #     run.update()

#----------
# wandb.config.update({'key':'value'},allow_val_change = True) 是往config中增加k-v pair
# 与run.config[key] = new_value 不同