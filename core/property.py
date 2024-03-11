import numpy as np 

class Property:
    def __init__(self, input_dim, input_constrs, output_dim, output_constrs):
        self.input_dim = input_dim
        self.input_constrs = input_constrs
        self.output_dim = output_dim
        self.output_constrs = output_constrs

    def generate(self, input_seed, output_seed, input_bound=2., output_bound=None):
        # Input
        np.random.seed(input_seed)
        self.P = np.round(np.random.uniform(low=-1, high=1, size=(self.input_dim, self.input_constrs)), 2)
        self.p = np.round(np.random.uniform(low=-1, high=1, size=(self.input_constrs,)), 2)
        if input_bound:
            lhs_bounds = np.concatenate((np.eye(self.input_dim), -np.eye(self.input_dim)), axis=1)
            rhs_bounds = np.array([input_bound]*(2*self.input_dim))
            self.P = np.concatenate((self.P, lhs_bounds), axis=1)
            self.p = np.concatenate((self.p, rhs_bounds))

        # Output
        np.random.seed(output_seed)
        self.R = np.round(np.random.uniform(low=-1, high=1, size=(self.output_dim, self.output_constrs)), 2)
        self.r = np.round(np.random.uniform(low=-1, high=1, size=(self.output_constrs,)), 2)
        if output_bound:
            lhs_bounds = np.concatenate((np.eye(self.output_dim), -np.eye(self.output_dim)), axis=1)
            rhs_bounds = np.array([output_bound]*(2*self.output_dim))
            self.R = np.concatenate((self.R, lhs_bounds), axis=1)
            self.r = np.concatenate((self.r, rhs_bounds))

        return self.P, self.p, self.R, self.r

    def print(self, poly_type='input'):
        if poly_type == 'input':
            if self.P.shape[0] == 1:
                for j in range(self.P.shape[1]):
                   print(f'\n{self.P[0,j]}x <= {self.p[j]}')
            elif self.P.shape[0] == 2:
                for j in range(self.P.shape[1]):
                   print(f'\n{self.P[0,j]}x + {self.P[1,j]}y <= {self.p[j]}')
            else:
                for j in range(self.P.shape[1]):
                    exp = ''
                    for i in range(self.P.shape[0]):
                        exp = exp + f' {self.P[i,j]}x_{i} +'
                    exp = exp[:-2]
                    print(f'\n{exp} <= {self.p[j]}')

        elif poly_type == 'output':
            if self.R.shape[0] == 1:
                for j in range(self.R.shape[1]):
                   print(f'\n{self.R[0,j]}x <= {self.r[j]}')
            elif self.R.shape[0] == 2:
                for j in range(self.R.shape[1]):
                   print(f'\n{self.R[0,j]}x + {self.R[1,j]}y <= {self.r[j]}')
            else:
                for j in range(self.R.shape[1]):
                    exp = ''
                    for i in range(self.R.shape[0]):
                        exp = exp + f' {self.R[i,j]}x_{i} +'
                    exp = exp[:-2]
                    print(f'\n{exp} <= {self.r[j]}')
