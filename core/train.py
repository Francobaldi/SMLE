import os
import pickle
import ast
from sklearn.metrics import mean_squared_error
import argparse
import datetime
from core.data import *
from core.property import *
from core.model import *


parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", type=int, default=2)
parser.add_argument("--output_dim", type=int, default=1)
parser.add_argument("--data_seed", type=int, required=True)
parser.add_argument("--input_seed", type=int, required=True)
parser.add_argument("--output_seed", type=int, required=True)
parser.add_argument("--input_constrs", type=int, required=False)
parser.add_argument("--output_constrs", type=int, required=False)
parser.add_argument("--architecture", type=str, required=True)
parser.add_argument("--model", type=str, default='smle')

args = parser.parse_args()
input_dim, output_dim = args.input_dim, args.output_dim 
data_seed = args.data_seed
input_seed, output_seed, input_constrs, output_constrs = args.input_seed, args.output_seed, args.input_constrs, args.output_constrs
architecture, model = ast.literal_eval(args.architecture), args.model

if not input_constrs:
    input_constrs = int(np.log2(input_dim) + 2)
if not output_constrs:
    output_constrs = int(np.log2(output_dim) + 2)


####################################################################################################
# Base
####################################################################################################
base = Base(architecture)


####################################################################################################
# Data
####################################################################################################
data_generator = Data(train_size=10000, test_size=1000, input_dim=input_dim, output_dim=output_dim)
x_train, y_train, x_test, y_test = data_generator.generate(seed=data_seed)


####################################################################################################
# Property
####################################################################################################
property_generator = Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)

print('\n================= PROPERTY ====================')
print('\n Input Poly') 
property_generator.print(poly_type='input')
print('\n Output Poly')
property_generator.print(poly_type='output')
print('\n===============================================')


####################################################################################################
# Training
####################################################################################################
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.system(f'mkdir {log_dir}')

if model == 'smle':
    lower_init = -1.
    upper_init = 1.

    model = SMLE(base=base, output_dim=output_dim, P=P, p=p, R=R, r=r, lower_init=lower_init, upper_init=upper_init, log_dir=log_dir)
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)

    log = {
         'gradients' : [],
         'weights' : [],
     }
 
    for f in os.listdir(log_dir):
         w = pickle.load(open(f'{log_dir}/{f}', 'rb'))
         for key in w.keys():
             log[key].append(w[key])

    log['input_dim'] = input_dim
    log['output_dim'] = output_dim
    log['data_seed'] = data_seed
    log['input_seed'] = input_seed
    log['output_seed'] = output_seed
    log['architecture'] = architecture
    log['P'] = P
    log['p'] = p
    log['R'] = R
    log['r'] = r
    log['train_membership'] = np.all(x_train @ P <= p, axis=1)
    log['test_membership'] = np.all(x_test @ P <= p, axis=1)
    log['train_intersection'] = np.sum(np.all(x_train @ P <= p, axis=1))/x_train.shape[0]
    log['test_intersection'] = np.sum(np.all(x_test @ P <= p, axis=1))/x_test.shape[0]
    log['final_weights'] = model.get_weights()
    log['loss'] = model.history.history['loss']
    log['val_loss'] = model.history.history['val_loss']
    log['y_test'] = y_test
    log['y_test_pred'] = model(x_test)
    log['mse_test'] = mean_squared_error(log['y_test'], log['y_test_pred'])
    violation = (log['y_test_pred'] @ R - r)[log['test_membership']]
    violation = np.sum(violation[violation > 0])
    log['violation_ontest'] = violation


    log_file = log_dir + '.pkl'
    pickle.dump(log, open(log_file, 'wb'))
    os.system(f'rm -rf {log_dir}')

elif model == 'naivesafe':
    model = Unsafe(base=base, output_dim=output_dim)
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)
    model = NaiveSafe(unsafe=model, P=P, p=p, R=R, r=r)

else:
    model = Unsafe(base=base, output_dim=output_dim)
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)

input_membership = np.all(x_test @ P <= p, axis=1)
violation = model(x_test) @ R - r
violation = violation[input_membership]
violation = np.sum(violation[violation > 0])
print(f'violation --> {violation}')
    
mse = mean_squared_error(y_test, model(x_test))
print(f'MSE --> {mse}')
