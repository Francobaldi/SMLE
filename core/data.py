import numpy as np 

class Data:
    def __init__(self, train_size, test_size, input_dim, output_dim):
        self.train_size = train_size
        self.test_size = test_size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def generate(self, seed):
        np.random.seed(seed)
        # Train Set
        x_train = np.random.uniform(low=-1, high=1, size=(self.train_size, self.input_dim))
        y_train = np.reshape(np.sum(x_train, axis=1), (self.train_size, 1))**np.arange(1, self.output_dim+1)
        # Test Set
        x_test = np.random.uniform(low=-1, high=1, size=(self.test_size, self.input_dim))
        y_test = np.reshape(np.sum(x_test, axis=1), (self.test_size, 1))**np.arange(1, self.output_dim+1)

        return x_train, y_train, x_test, y_test
