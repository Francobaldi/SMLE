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
name = 'dynamic-box'
res_dir = f'experiments/{name}/results'
agg_res_dir = f'experiments/{name}/aggregated_results'


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


frame = pd.DataFrame({
    'model' : [model.split('_')[0] for model in results.keys()],
    'dimension' : [str((int(model.split('_')[1]), int(model.split('_')[2]))) for model in results.keys()],
    'property' : [str((int(model.split('_')[3]), int(model.split('_')[4]))) if 'unsafe' not in model else (-1, -1) for model in results.keys()],
    'seed' : [int(model.split('_')[3]) if 'unsafe' in model else int(model.split('_')[5]) for model in results.keys()],
    'difficulty_ontrain' : [results[model]['difficulty_ontrain'] for model in results.keys()],
    'difficulty_ontest' : [results[model]['difficulty_ontest'] for model in results.keys()],
    'mse_test' : [results[model]['mse_test'] for model in results.keys()],
    }, index = list(results.keys()))

frame['difficulty_ontrain'] = (frame['difficulty_ontrain']/10000).round(2)
frame['difficulty_ontest'] = (frame['difficulty_ontest']/1000).round(2)
frame = frame.sort_values(by=['dimension', 'difficulty_ontrain', 'difficulty_ontest'])
frame = frame.groupby(['model', 'dimension', 'property']).mean().reset_index().drop('seed', axis=1)
frame = frame[frame['model'] != 'unsafe']
frame = frame.pivot(index=['dimension', 'property', 'difficulty_ontrain', 'difficulty_ontest'], columns='model', values='mse_test').reset_index()

previous_frame = pd.read_csv(f'experiments/{name}/synthetic_experiment_results.csv')
previous_frame = previous_frame.groupby(['model', 'dimension', 'property']).mean().reset_index().drop('seed', axis=1)
previous_frame = previous_frame[previous_frame['model'] != 'unsafe']
previous_frame = previous_frame.pivot(index=['dimension', 'property', 'difficulty_ontrain', 'difficulty_ontest'], columns='model', values='mse_test').reset_index()

frame = pd.merge(previous_frame, frame, on=['dimension', 'property', 'difficulty_ontrain', 'difficulty_ontest'], suffixes=('_static', '_dynamic'))
frame['relative_static'] = frame['smle_static']/frame['naivesafe_static']
frame['relative_dynamic'] = frame['smle_dynamic']/frame['naivesafe_dynamic']


####################################################################
# MSE Plot by Dimension-Property
####################################################################
for dim in frame['dimension'].unique():
    f = frame.loc[frame['dimension'] == dim]
    f = f.groupby(['difficulty_ontest'])[['relative_static', 'relative_dynamic']].mean().reset_index()
    plt.figure(figsize=(20, 10))
    sns.barplot(data=f, x='difficulty_ontest', y='relative_static', label='static') 
    sns.barplot(data=f, x='difficulty_ontest', y='relative_dynamic', label='dynamic') 
    dim = (int(dim.split(',')[0].split('(')[1]), int(dim.split(',')[1].split(')')[0]))
    plt.title(f'{dim[0]}x{dim[1]}')
    plt.ylabel('relative MSE')
    plt.legend()
    #plt.show()
    plt.savefig(f'{agg_res_dir}/bar_{dim[0]}_{dim[1]}.pdf', bbox_inches='tight')


####################################################################
# MSE Plot by Dimension
####################################################################
f = frame.groupby(['dimension'])[['relative_static', 'relative_dynamic']].mean().reset_index()
plt.figure(figsize=(20, 10))
sns.barplot(data=f, x='dimension', y='relative_static', label='static') 
sns.barplot(data=f, x='dimension', y='relative_dynamic', label='dynamic') 
plt.yticks([2.5*i for i in range(11)])
plt.ylabel('relative MSE')
plt.legend()
#plt.show()
plt.savefig(f'{agg_res_dir}/bar.pdf', bbox_inches='tight')
