import os
import sys
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from core.optimization import *

####################################################################################################
# Base
####################################################################################################
class Base(keras.Model):
    def __init__(self, architecture):
        super().__init__()
        self._layers = [keras.layers.Dense(units=units, activation=activation) for (units, activation) in architecture] 
        self.architecture = architecture

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


####################################################################################################
# Unsafe
####################################################################################################
class Unsafe(keras.Model):
    def __init__(self, base, output_dim):
        super().__init__()
        self.base = base
        self.g = keras.layers.Dense(units=output_dim, activation='linear')

    def call(self, inputs):
        return self.g(self.base(inputs))


####################################################################################################
# Naive
####################################################################################################
class NaiveSafe(keras.Model):
    def __init__(self, unsafe, P, p, R, r):
        super().__init__()
        self.P = P
        self.p = p
        self.R = R
        self.r = r

        self.OP = OutputProjection(R=R, r=r)

        self.unsafe = unsafe
        self.unsafe.trainable = False


    def call(self, inputs):

        outputs = []
        for i in inputs:
            i = tf.expand_dims(i, 0)
            o = self.unsafe(i)
            
            input_membership = tf.reduce_all(i @ self.P <= self.p, axis=1)
            output_membership = tf.reduce_all(o @ self.R <= self.r, axis=1)
            if input_membership and not output_membership:
                o = self.OP.project(tf.reshape(o, shape=(-1)).numpy())
                o = tf.cast(tf.expand_dims(o, 0), tf.float32)

            outputs.append(o)
        outputs = tf.concat(outputs, axis=0)

        return outputs



####################################################################################################
# Safe
####################################################################################################
class LowerClip(keras.layers.Layer):
    def __init__(self, hidden_dim, lower_init, **kwargs):
         super().__init__(**kwargs)
         self.hidden_dim = hidden_dim
         self.lower_init = lower_init

    def build(self, input_shape):
         self.B_lower = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='zeros', trainable=True)
         self.b_lower = self.add_weight(shape=(self.hidden_dim,), initializer=self.lower_init, trainable=True)

    def call(self, inputs):
         return inputs @ self.B_lower + self.b_lower


class UpperClip(keras.layers.Layer):
    def __init__(self, hidden_dim, upper_init, **kwargs):
         super().__init__(**kwargs)
         self.hidden_dim = hidden_dim
         self.upper_init = upper_init

    def build(self, input_shape):
         self.B_upper = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='zeros', trainable=True)
         self.b_upper = self.add_weight(shape=(self.hidden_dim,), initializer=self.upper_init, trainable=True)

    def call(self, inputs):
         return inputs @ self.B_upper + self.b_upper


class SMLE(keras.Model):
    def __init__(self, base, output_dim, P, p, R, r, lower_init=-1., upper_init=1., log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.P = P 
        self.p = p 
        self.R = R 
        self.r = r 
        self.lower_init = tf.constant_initializer(lower_init)
        self.upper_init = tf.constant_initializer(upper_init)

        self.CE = CounterExample(P=P, p=p, R=R, r=r)
        self.WP = WeightProjection(R=R, r=r)

        self.base = base
        self.g = keras.layers.Dense(output_dim, name='g')
        self.lower_clip = LowerClip(hidden_dim=base.architecture[-1][0], lower_init=self.lower_init, name='lower_clip')
        self.upper_clip = UpperClip(hidden_dim=base.architecture[-1][0], upper_init=self.upper_init, name='upper_clip')
        self.g_poly = keras.layers.Dense(output_dim, name='g_poly')


    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        # Update
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        #################### Counter-Example Searching #########################
        B_lower, b_lower = self.lower_clip.get_weights()
        B_upper, b_upper = self.upper_clip.get_weights()
        W, w = self.g_poly.get_weights()
        y, u = self.CE.generate(B_lower=B_lower, b_lower=b_lower, B_upper=B_upper, b_upper=b_upper, W=W, w=w)
        ########################################################################

        ######################## Weight Projection #############################
        violation = np.sum(u[u > 0])
        if violation > 0:
            W, w = self.WP.project(y=y, W=W, w=w)   
            self.g_poly.set_weights([W, w])
        ########################################################################
        
        ########################## Process Monitor #############################
        if self.log_dir:
            epoch = len(self.history.epoch)
            filename = f'{self.log_dir}/{epoch}.pkl'

            if not os.path.isfile(filename):
                log = {
                  'gradients' : [g.numpy() for g in gradients],
                  'weights' : self.get_weights()
                }
                pickle.dump(log, open(filename, 'wb'))
        ########################################################################
        
        ############################# Logging ##################################
        sys.stdout.write(f'\r{80*" "}violation --> {violation}{5*" "}projection --> {violation > 0}')
        sys.stdout.flush()
        ########################################################################

        # Metric update
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


    def call(self, inputs):
        poly_membership = tf.expand_dims(tf.reduce_all(inputs @ self.P <= self.p, axis=1), axis=1)
        outputs = tf.where(poly_membership, 
                          self.g_poly(tf.maximum(tf.minimum(self.base(inputs), self.upper_clip(inputs)), self.lower_clip(inputs))), 
                          self.g(self.base(inputs)))

        return outputs
