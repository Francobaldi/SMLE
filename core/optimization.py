import pyomo.environ as pyo
import numpy as np


class CounterExample:
    def __init__(self, P, p, R, r):
        self.P = P
        self.p = p
        self.R = R
        self.r = r

    def generate(self, B_lower, b_lower, B_upper, b_upper, W, w):
        self.model = pyo.ConcreteModel()

        # Variables
        self.model.x = pyo.Var(range(self.P.shape[0]))
        self.model.y = pyo.Var(range(W.shape[0]))
        self.model.z = pyo.Var(range(W.shape[1]))
        self.model.u = pyo.Var(range(self.R.shape[1]))
        self.model.b = pyo.Var(range(self.R.shape[1]), domain=pyo.Binary)

        # Constraints
        @self.model.Constraint(range(self.P.shape[1]))
        def in_poly(model, j):
            return self.model.x @ self.P[:,j] <= self.p[j]
        @self.model.Constraint(range(W.shape[0]))
        def lower_clip(model, j):
            return self.model.y[j] >= self.model.x @ B_lower[:,j] + b_lower[j] 
        @self.model.Constraint(range(W.shape[0]))
        def upper_clip(model, j):
            return self.model.y[j] <= self.model.x @ B_upper[:,j] + b_upper[j] 
        @self.model.Constraint(range(W.shape[1]))
        def g(model, j):
            return self.model.z[j] == self.model.y @ W[:,j] + w[j]
        @self.model.Constraint(range(self.R.shape[1]))
        def violation(model, j):
            return self.model.u[j] == self.model.z @ self.R[:,j] - self.r[j]

        # Objective
        self.model.objective = pyo.Objective(expr=sum(self.model.b[j] * self.model.u[j] for j in range(self.R.shape[1])), sense=pyo.maximize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        y = np.array([self.model.y[j].value for j in range(W.shape[0])])
        u = np.array([self.model.u[j].value for j in range(self.R.shape[1])])

        return y, u

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})
    

class WeightProjection:
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def project(self, y, W, w):
        self.model = pyo.ConcreteModel()

        # Variables
        self.model.W = pyo.Var(range(W.shape[0]), range(W.shape[1]))
        self.model.w = pyo.Var(range(W.shape[1]))
        self.model.z = pyo.Var(range(W.shape[1]))

        # Constraints
        @self.model.Constraint(range(W.shape[1]))
        def g(model, j):
            return self.model.z[j] == sum(y[i] * self.model.W[i,j] for i in range(W.shape[0])) + self.model.w[j] 
        @self.model.Constraint(range(self.R.shape[1]))
        def property(model, j):
            return self.model.z @ self.R[:,j] <= self.r[j]

        # Objective
        expr = sum(sum((W - self.model.W)**2))
        expr += sum((w - self.model.w)**2)
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        W = np.array([[self.model.W[i,j].value for j in range(W.shape[1])] for i in range(W.shape[0])])
        w = np.array([self.model.w[j].value for j in range(w.shape[0])])

        return W, w

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})


class OutputProjection:
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def project(self, z):
        self.model = pyo.ConcreteModel()

        # Variables
        self.model.z = pyo.Var(range(self.R.shape[0]))

        # Constraints
        @self.model.Constraint(range(self.R.shape[1]))
        def property(model, j):
            return self.model.z @ self.R[:,j] <= self.r[j]

        # Objective
        expr = sum((z - self.model.z)**2)
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        z = np.array([self.model.z[j].value for j in range(self.R.shape[0])])

        return z

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})
