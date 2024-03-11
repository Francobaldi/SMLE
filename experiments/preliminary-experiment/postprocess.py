import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn 

#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'preliminary-experiment'
input_dim = 2
output_dim = 1
input_constrs = 3
output_constrs = 1
input_seeds = range(3)
output_seeds = range(3)
train_size = 10000
test_size = 1000
optimizer = 'adam'
loss = 'MSE'
batch_size = 64
epochs = 10
validation_split = 0.2
data_seed = 0
train_seeds = range(2)
hidden_dims = [4,8,16,32]
res_dir = f'experiments/{name}/results'
agg_res_dir = f'experiments/{name}/aggregated_results'


#######################################################################################################
# Results Extraction
#######################################################################################################
models = [f'baseline_{hidden_dim}' for hidden_dim in hidden_dims]
models = models + [f'smls_{hidden_dim}_{input_seed}_{output_seed}'for hidden_dim in hidden_dims for input_seed in input_seeds for output_seed in output_seeds]

results = {}
for model in models:
    results[model] = {
            'loss' : [],
            'mse' : [],
            'train_intersection' : [],
            }

for model in models: 
    for train_seed in train_seeds:
        w = pickle.load(open(f'{res_dir}/{model.split("_")[0]}_{train_seed}_{"_".join(model.split("_")[1:])}.pkl', 'rb'))
        results[model]['loss'] += [w['loss']]
        results[model]['mse'] += [w['mse']]
        results[model]['train_intersection'] += [w['train_intersection']]
    results[model]['loss'] = np.mean(results[model]['loss'], axis=0)
    results[model]['mse'] = np.mean(results[model]['mse'], axis=0)
    if None in results[model]['train_intersection']:
        results[model]['train_intersection'] = None
    else:
        results[model]['train_intersection'] = np.mean(results[model]['train_intersection'], axis=0)
        

####################################################################
# MSE
####################################################################
plt.figure(figsize=(16, 12))
plt.barh(models, [results[model]['mse'] for model in models], color='skyblue')
plt.xlim(0, 1.0)
plt.xlabel('MSE')
#plt.show()
plt.savefig(f'{agg_res_dir}/mse.pdf', bbox_inches='tight')
plt.clf()


####################################################################
# Filter Outliers & Aggregate by Hidden
####################################################################
results = {key : results[key] for key in results.keys() if '0_2' not in key} 

agg_results = {}
for hidden_dim in hidden_dims:
    agg_results[f'baseline_{hidden_dim}'] = results[f'baseline_{hidden_dim}']
    agg_results[f'smls_{hidden_dim}'] = {}
    agg_results[f'smls_{hidden_dim}']['mse'] = np.mean([results[model]['mse'] for model in results.keys() if f'smls_{hidden_dim}' in model])
    agg_results[f'smls_{hidden_dim}']['loss'] = np.mean([results[model]['loss'] for model in results.keys() if f'smls_{hidden_dim}' in model], axis=0)


####################################################################
# Loss by Hidden_Dim
####################################################################
plt.figure(figsize=(12, 6))
for model in agg_results.keys():
    plt.plot(range(epochs), agg_results[model]['loss'], label=model)
plt.legend()
#plt.show()
plt.savefig(f'{agg_res_dir}/loss_by_dim.pdf', bbox_inches='tight')
plt.clf()


####################################################################
# MSE by Hidden_Dim
####################################################################
agg_results = pd.DataFrame(
        {'baseline' : [agg_results[f'baseline_{hidden_dim}']['mse'] for hidden_dim in hidden_dims],
         'smls' : [agg_results[f'smls_{hidden_dim}']['mse'] for hidden_dim in hidden_dims]},
        index = hidden_dims)

agg_results.plot.bar(rot=0, figsize=(12, 8))
plt.ylabel('MSE')
#plt.show()
plt.savefig(f'{agg_res_dir}/mse_by_dim.pdf', bbox_inches='tight')
plt.clf()
