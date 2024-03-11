import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.color_palette('colorblind')


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'synthetic-experiment'
res_dir = f'experiments/{name}/results'
agg_res_dir = f'experiments/{name}/aggregated_results'
dimensions = [(input_dim, output_dim) for input_dim in [2,4,8,16] for output_dim in [2,4,8,16]]
dimensions = [dimension for dimension in dimensions if dimension != (16,16)]
train_seeds = range(2)
properties = [(input_seed, output_seed) for input_seed in [33,590,994,487] for output_seed in [510,743,980,863]]


#######################################################################################################
# Results Extraction
#######################################################################################################
results = {}
for model in os.listdir(res_dir):
    w = pickle.load(open(f'{res_dir}/{model}', 'rb'))
    model = model.split('.')[0]
    results[model] = {
        'loss' : w['loss'],
        'input_train_membership' : w['input_train_membership'],
        'output_train_membership' : w['output_train_membership'],
        'input_test_membership' : w['input_test_membership'],
        'output_test_membership' : w['output_test_membership'],
        'final_weights' : w['final_weights'],
        'mse_test' : w['mse_test'],
        }
    if w['input_train_membership'] is not None:
        results[model]['difficulty_ontrain'] = np.sum(w['input_train_membership'] & ~w['output_train_membership'])
        results[model]['difficulty_ontest'] = np.sum(w['input_test_membership'] & ~w['output_test_membership'])
    else:
        results[model]['difficulty_ontrain'] = 0.
        results[model]['difficulty_ontest'] = 0.
#   if 'smle' in model:
#       results[model]['loss'] = results[model]['loss'][::2]


frame = pd.DataFrame({
    'model' : [model.split('_')[0] for model in results.keys()],
    'dimension' : [(int(model.split('_')[1]), int(model.split('_')[2])) for model in results.keys()],
    'property' : [(int(model.split('_')[3]), int(model.split('_')[4])) if 'unsafe' not in model else (-1, -1) for model in results.keys()],
    'seed' : [int(model.split('_')[3]) if 'unsafe' in model else int(model.split('_')[5]) for model in results.keys()],
    'difficulty_ontrain' : [results[model]['difficulty_ontrain'] for model in results.keys()],
    'difficulty_ontest' : [results[model]['difficulty_ontest'] for model in results.keys()],
    'mse_test' : [results[model]['mse_test'] for model in results.keys()],
    }, index = list(results.keys()))

frame['difficulty_ontrain'] = (frame['difficulty_ontrain']/10000).round(2)
frame['difficulty_ontest'] = (frame['difficulty_ontest']/1000).round(2)
frame = frame.sort_values(by=['dimension', 'difficulty_ontrain', 'difficulty_ontest'])

best = frame.loc[frame.groupby(['model', 'dimension', 'property'])['mse_test'].idxmin()]

frame = frame.groupby(['model', 'dimension', 'property']).mean().reset_index().drop('seed', axis=1)
for dim in dimensions:
    unsafe_mse = frame.loc[(frame['dimension'] == dim) & (frame['model'] == 'unsafe'), 'mse_test'].iloc[0]
    frame.loc[frame['dimension'] == dim, 'mse_test'] /= unsafe_mse
frame = frame[frame['model'] != 'unsafe']


####################################################################
# MSE Plot by Dimension-Property
####################################################################
for dim in dimensions:
    f = frame.loc[frame['dimension'] == dim]
    f = f.groupby(['difficulty_ontest', 'model'])['mse_test'].mean().reset_index()
    plt.figure(figsize=(20, 10))
    sns.barplot(data=f, x='difficulty_ontest', y='mse_test', hue='model') 
    plt.title(f'{dim[0]}x{dim[1]}')
    plt.legend(loc='upper left')
#   plt.show()
    plt.savefig(f'{agg_res_dir}/bar_{dim[0]}_{dim[1]}.pdf', bbox_inches='tight')


####################################################################
# MSE Plot by Dimension
####################################################################
f = frame.groupby(['dimension', 'model'])['mse_test'].mean().reset_index()
f['dimension'] = [f'{input_dim}x{output_dim}' for input_dim, output_dim in f['dimension']]
plt.figure(figsize=(20, 10))
sns.barplot(data=f, x='dimension', y='mse_test', hue='model') 
plt.yticks([2.5*i for i in range(11)])
plt.legend(loc='upper left')
#plt.show()
plt.savefig(f'{agg_res_dir}/bar.pdf', bbox_inches='tight')


####################################################################
# Loss
####################################################################
for dim in dimensions:
    plt.figure(figsize=(20, 10))
    unsafe = np.mean((results[f'unsafe_{dim[0]}_{dim[1]}_0']['loss'], results[f'unsafe_{dim[0]}_{dim[1]}_1']['loss']), axis=0)
    plt.plot(range(len(unsafe)), unsafe, label='unsafe')
    for prop in frame.loc[frame['dimension'] == dim, 'property'].unique():
        smle = np.mean((results[f'smle_{dim[0]}_{dim[1]}_{prop[0]}_{prop[1]}_0']['loss'], results[f'smle_{dim[0]}_{dim[1]}_{prop[0]}_{prop[1]}_1']['loss']), axis=0)
        plt.plot(range(len(smle)), smle, label=f'smle {prop}')
    plt.title(f'{dim[0]}x{dim[1]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    plt.savefig(f'{agg_res_dir}/loss_{dim[0]}_{dim[1]}.pdf', bbox_inches='tight')
