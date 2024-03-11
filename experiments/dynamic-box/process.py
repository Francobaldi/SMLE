import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from core.data import *
from core.model import *
from core.property import *
from core.optimization import *


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'dynamic-box'
train_size = 10000
test_size = 1000
optimizer = 'adam'
loss = 'mse'
batch_size = 128
validation_split = 0.2
data_seed = 0
res_dir = f'experiments/{name}/results'
train_seeds = range(2)


####################################################################
# Identify Relevant Cases from Syntethic Experiment Results
####################################################################
frame = pd.read_csv(f'experiments/{name}/synthetic_experiment_results.csv')
frame = frame.groupby(['model', 'dimension', 'property']).mean().reset_index().drop('seed', axis=1)
frame = frame[frame['model'] != 'unsafe']

for dim in frame['dimension'].unique():
    for prop in frame.loc[frame['dimension'] == dim, 'property'].unique():
        naivsafe_mse = frame.loc[(frame['dimension'] == dim) & (frame['property'] == prop) & (frame['model'] == 'naivesafe'), 'mse_test'].iloc[0]
        frame.loc[(frame['dimension'] == dim) & (frame['property'] == prop), 'mse_test'] /= naivsafe_mse
frame = frame[frame['model'] != 'naivesafe']
frame = frame[frame['mse_test'] >= 5]

dimensions = [(int(dim.split(',')[0].split('(')[1]), int(dim.split(',')[1].split(')')[0])) for dim in frame['dimension'].unique()]
properties = {dim : [(int(prop.split(',')[0].split('(')[1]), int(prop.split(',')[1].split(')')[0])) for prop in frame.loc[frame['dimension']==str(dim), 'property'].unique()] for dim in dimensions}


#######################################################################################################
# Process
#######################################################################################################
for (input_dim, output_dim) in dimensions:
    input_constrs = int(np.log2(input_dim) + 2)
    output_constrs = int(np.log2(output_dim) + 2)
    architecture = [(input_dim*output_dim, 'relu'), (2*input_dim*output_dim, 'relu'), (input_dim*output_dim, 'relu')]
    epochs = int(6*input_dim*output_dim/np.log2(input_dim*output_dim))
    base = Base(architecture)


    ###########################################################################################################
    # Generators
    ###########################################################################################################
    data_generator = Data(train_size=train_size, test_size=test_size, input_dim=input_dim, output_dim=output_dim)
    property_generator = Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)


    ###########################################################################################################
    # Data
    ###########################################################################################################
    x_train, y_train, x_test, y_test = data_generator.generate(seed=data_seed)


    ###########################################################################################################
    # Training
    ###########################################################################################################
    for train_seed in train_seeds:
        print(f'\ninput_dim --> {input_dim} -- output_dim --> {output_dim} -- train seed --> {train_seed}')  

        ###########################################################################################################
        # Unsafe
        ###########################################################################################################
        print('=============================================================================')
        print('Unsafe')
        print('=============================================================================')
        log_dir = f'{res_dir}/unsafe_{input_dim}_{output_dim}_{train_seed}'
        log_file = log_dir + '.pkl'

        os.system(f'mkdir {log_dir}')

        model = Unsafe(base=base, output_dim=output_dim)
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

        log = {}
        log['dimension'] = (input_dim, output_dim)
        log['data_seed'] = data_seed
        log['property'] = None
        log['train_seed'] = train_seed
        log['P'] = None
        log['p'] = None
        log['R'] = None
        log['r'] = None
        log['input_train_membership'] = None
        log['input_test_membership'] = None
        log['output_train_membership'] = None
        log['output_test_membership'] = None
        log['loss'] = model.history.history['loss']
        log['val_loss'] = model.history.history['val_loss']
        log['y_test'] = y_test
        log['y_test_pred'] = model(x_test)
        log['mse_test'] = mean_squared_error(y_test, model(x_test))
        log['final_weights'] = model.get_weights()

        print(f'MSE --> {log["mse_test"]}')

        unsafe = model
        pickle.dump(log, open(log_file, 'wb'))
        os.system(f'rm -rf {log_dir}')

        for (input_seed, output_seed) in properties[(input_dim, output_dim)]:
            ###########################################################################################################
            # Polytope
            ###########################################################################################################
            P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)

            print('\n================= PROPERTY ====================')
            print('\n Input Poly')
            property_generator.print(poly_type='input')
            print('\n Output Poly')
            property_generator.print(poly_type='output')
            print('\n===============================================')


            ###########################################################################################################
            # Naive-Safe
            ###########################################################################################################
            print('=============================================================================')
            print('Naive-Safe')
            print('=============================================================================')
            log_dir = f'{res_dir}/naivesafe_{input_dim}_{output_dim}_{input_seed}_{output_seed}_{train_seed}'
            log_file = log_dir + '.pkl'

            if not os.path.isfile(log_file):
                os.system(f'mkdir {log_dir}')
                
                model = NaiveSafe(unsafe=unsafe, P=P, p=p, R=R, r=r)

                log = {}
                log['dimension'] = (input_dim, output_dim)
                log['data_seed'] = data_seed
                log['property'] = (input_seed, output_seed)
                log['train_seed'] = train_seed
                log['P'] = P
                log['p'] = p
                log['R'] = R
                log['r'] = r
                log['input_train_membership'] = np.all(x_train @ P <= p, axis=1)
                log['input_test_membership'] = np.all(x_test @ P <= p, axis=1)
                log['output_train_membership'] = np.all(y_train @ R <= r, axis=1)
                log['output_test_membership'] = np.all(y_test @ R <= r, axis=1)
                log['loss'] = None
                log['val_loss'] = None
                log['y_test'] = y_test
                log['y_test_pred'] = model(x_test)
                log['mse_test'] = mean_squared_error(y_test, model(x_test))
                log['final_weights'] = model.get_weights()

                print(f'MSE --> {log["mse_test"]}')

                pickle.dump(log, open(log_file, 'wb'))
                os.system(f'rm -rf {log_dir}')

            #######################################################################################################
            # SMLE
            #######################################################################################################
            print('=============================================================================')
            print('SMLE')
            print('=============================================================================')
            log_dir = f'{res_dir}/smle_{input_dim}_{output_dim}_{input_seed}_{output_seed}_{train_seed}'
            log_file = log_dir + '.pkl'

            if not os.path.isfile(log_file):
                os.system(f'mkdir {log_dir}')

                lower_init = -1.
                upper_init = 1.

                model = SMLE(base=base, output_dim=output_dim, P=P, p=p, R=R, r=r, lower_init=lower_init, upper_init=upper_init, log_dir=log_dir)
                model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
                model.fit(x_train, y_train, batch_size=batch_size, epochs=2*epochs, validation_split=validation_split)

                log = {
                     'gradients' : [],
                     'weights' : [],
                 }

                for f in os.listdir(log_dir):
                     w = pickle.load(open(f'{log_dir}/{f}', 'rb'))
                     for key in w.keys():
                         log[key].append(w[key])

                log['dimension'] = (input_dim, output_dim)
                log['data_seed'] = data_seed
                log['property'] = (input_seed, output_seed)
                log['train_seed'] = train_seed
                log['P'] = P
                log['p'] = p
                log['R'] = R
                log['r'] = r
                log['input_train_membership'] = np.all(x_train @ P <= p, axis=1)
                log['input_test_membership'] = np.all(x_test @ P <= p, axis=1)
                log['output_train_membership'] = np.all(y_train @ R <= r, axis=1)
                log['output_test_membership'] = np.all(y_test @ R <= r, axis=1)
                log['loss'] = model.history.history['loss']
                log['val_loss'] = model.history.history['val_loss']
                log['y_test'] = y_test
                log['y_test_pred'] = model(x_test)
                log['mse_test'] = mean_squared_error(y_test, model(x_test))
                log['final_weights'] = model.get_weights()

                print(f'MSE --> {log["mse_test"]}')

                pickle.dump(log, open(log_file, 'wb'))
                os.system(f'rm -rf {log_dir}')
