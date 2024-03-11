import os
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
name = 'synthetic-experiment'
train_size = 10000
test_size = 1000
optimizer = 'adam'
loss = 'mse'
batch_size = 128
validation_split = 0.2
data_seed = 0
res_dir = f'experiments/{name}/results'
dimensions = [(input_dim, output_dim) for input_dim in [2,4,8,16] for output_dim in [2,4,8,16]]
train_seeds = range(2)
n_properties = 4

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

    
    #######################################################################################################
    # Properties
    #######################################################################################################
    input_intersections = []
    output_intersections = []
    for seed in range(1000):
        P, p, R, r = property_generator.generate(input_seed=seed, output_seed=seed)
        input_intersections.append((seed, np.sum(np.all(x_train @ P <= p, axis=1))/x_train.shape[0]))
        output_intersections.append((seed, np.sum(np.all(y_train @ R <= r, axis=1))/y_train.shape[0]))
    input_intersections = np.array(input_intersections, dtype=[('seed', int), ('rate', float)])
    input_intersections = np.flip(np.sort(input_intersections, order='rate'))
    output_intersections = np.array(output_intersections, dtype=[('seed', int), ('rate', float)])
    output_intersections = np.flip(np.sort(output_intersections, order='rate'))
    input_intersections = np.array(input_intersections[input_intersections['rate'] > 0])
    output_intersections = np.array(output_intersections[output_intersections['rate'] > 0])

    # Select polytopes based on a geometric sequence of their intersection rate, highest to lowest
    select = (np.geomspace(1, len(input_intersections), num=n_properties, endpoint=True, dtype=int, axis=0) - 1).astype(int)
    input_seeds = input_intersections['seed'][select]
    select = (np.geomspace(1, len(output_intersections), num=n_properties, endpoint=True, dtype=int, axis=0) - 1).astype(int)
    output_seeds = output_intersections['seed'][select]
    properties = [(input_seed, output_seed) for input_seed in input_seeds for output_seed in output_seeds]


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
        log['dimensions'] = (input_dim, output_dim)
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
        log['lower_clip'] = None
        log['upper_clip'] = None
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

        for (input_seed, output_seed) in properties:
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
                log['dimensions'] = (input_dim, output_dim)
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
                log['lower_clip'] = None
                log['upper_clip'] = None
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

                lastlayer_dim = architecture[-1][0]
                lower_clip = [-1.]*lastlayer_dim
                upper_clip = [1.]*lastlayer_dim

                model = SMLE(base=base, output_dim=output_dim, P=P, p=p, R=R, r=r, lower_clip=lower_clip, upper_clip=upper_clip, log_dir=log_dir)
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

                log['dimensions'] = (input_dim, output_dim)
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
                log['lower_clip'] = lower_clip
                log['upper_clip'] = upper_clip
                log['loss'] = model.history.history['loss']
                log['val_loss'] = model.history.history['val_loss']
                log['y_test'] = y_test
                log['y_test_pred'] = model(x_test)
                log['mse_test'] = mean_squared_error(y_test, model(x_test))
                log['final_weights'] = model.get_weights()

                print(f'MSE --> {log["mse_test"]}')

                pickle.dump(log, open(log_file, 'wb'))
                os.system(f'rm -rf {log_dir}')
