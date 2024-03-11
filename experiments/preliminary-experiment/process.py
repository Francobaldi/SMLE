import os
import time
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from core.data import *
from core.model import *
from core.property import *
from core.optimization import *


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
batch_size = 128
epochs = 10
validation_split = 0.2
data_seed = 0
train_seeds = range(2)
hidden_dims = [4,8,16,32]
res_dir = f'experiments/{name}/results'

###########################################################################################################
# Generators
###########################################################################################################
data_generator = Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
property_generator = Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)

###########################################################################################################
# Data
###########################################################################################################
x_train, y_train, x_test, y_test = data_generator.generate(seed=data_seed)

###########################################################################################################
# Training
###########################################################################################################
for train_seed in train_seeds:
    for hidden_dim in hidden_dims:
        print(f'\ntrain seed --> {train_seed} -- hidden dim --> {hidden_dim}')  

        ###########################################################################################################
        # Baseline
        ###########################################################################################################
        log_dir = f'{res_dir}/baseline_{train_seed}_{hidden_dim}'
        log_file = log_dir + '.pkl'

        if not os.path.isfile(log_file):
            os.system(f'mkdir {log_dir}')

            model = Baseline(hidden_dim=hidden_dim, output_dim=output_dim)
            model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

            log = {}
            log['input_dim'] = input_dim
            log['hidden_dim'] = hidden_dim
            log['output_dim'] = output_dim
            log['data_seed'] = data_seed
            log['input_seed'] = None
            log['output_seed'] = None
            log['train_seed'] = train_seed
            log['P'] = None
            log['p'] = None
            log['R'] = None
            log['r'] = None
            log['train_membership'] = None
            log['test_membership'] = None
            log['train_intersection'] = None
            log['test_intersection'] = None
            log['lower_clip'] = None
            log['upper_clip'] = None
            log['loss'] = model.history.history['loss']
            log['val_loss'] = model.history.history['val_loss']
            log['y_test'] = y_test
            log['y_test_pred'] = model(x_test)
            log['mse'] = mean_squared_error(y_test, model(x_test))
            log['final_weights'] = model.get_weights()

            pickle.dump(log, open(log_file, 'wb'))
            os.system(f'rm -rf {log_dir}')


        for input_seed in input_seeds:
            for output_seed in output_seeds:
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


                #######################################################################################################
                # SMLS
                #######################################################################################################
                log_dir = f'{res_dir}/smls_{train_seed}_{hidden_dim}_{input_seed}_{output_seed}'
                log_file = log_dir + '.pkl'

                if not os.path.isfile(log_file):
                    os.system(f'mkdir {log_dir}')

                    lower_clip = [-1.]*hidden_dim
                    upper_clip = [1.]*hidden_dim

                    model = SMLS(hidden_dim=hidden_dim, output_dim=output_dim, P=P, p=p, R=R, r=r, lower_clip=lower_clip, upper_clip=upper_clip, log_dir=log_dir)
                    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
                    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

                    log = {
                         'W_f_grad' : [],
                         'w_f_grad' : [],
                         'W_g_grad' : [],
                         'w_g_grad' : [],
                         'W_g_poly_grad' : [],
                         'w_g_poly_grad' : [],
                         'W_f' : [],
                         'w_f' : [],
                         'W_g' : [],
                         'w_g' : [],
                         'W_g_poly' : [],
                         'w_g_poly' : []
                     }

                    for f in os.listdir(log_dir):
                         w = pickle.load(open(f'{log_dir}/{f}', 'rb'))
                         for key in w.keys():
                             log[key].append(w[key])

                    log['input_dim'] = input_dim
                    log['hidden_dim'] = hidden_dim
                    log['output_dim'] = output_dim
                    log['data_seed'] = data_seed
                    log['input_seed'] = input_seed
                    log['output_seed'] = output_seed
                    log['train_seed'] = train_seed
                    log['P'] = P
                    log['p'] = p
                    log['R'] = R
                    log['r'] = r
                    log['train_membership'] = np.all(x_train @ P <= p, axis=1)
                    log['test_membership'] = np.all(x_test @ P <= p, axis=1)
                    log['train_intersection'] = np.sum(np.all(x_train @ P <= p, axis=1))/x_train.shape[0]
                    log['test_intersection'] = np.sum(np.all(x_test @ P <= p, axis=1))/x_test.shape[0]
                    log['lower_clip'] = lower_clip
                    log['upper_clip'] = upper_clip
                    log['loss'] = model.history.history['loss']
                    log['val_loss'] = model.history.history['val_loss']
                    log['y_test'] = y_test
                    log['y_test_pred'] = model(x_test)
                    log['mse'] = mean_squared_error(y_test, model(x_test))
                    log['final_weights'] = model.get_weights()

                    pickle.dump(log, open(log_file, 'wb'))
                    os.system(f'rm -rf {log_dir}')
