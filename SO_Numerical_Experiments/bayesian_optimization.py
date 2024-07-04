import torch
import os
import pandas as pd
from botorch.test_functions import Ackley
from botorch.test_functions import Levy
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize, standardize
from torch.quasirandom import SobolEngine
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement

class BayesianOptimization:
    def __init__(self, fun, dim=5, batch_size=10, n_init=None, 
                epochs=10, num_restarts=10, raw_samples=512, seed=0, 
                bound=32.768, acqf_type='qEI'):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.dim = dim
        self.batch_size = batch_size
        self.n_init = n_init if n_init is not None else 2 * dim
        self.epochs = epochs
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.seed = seed
        self.bound = bound
        self.acqf_type = acqf_type
        
        self.fun = fun
        self.fun.bounds[0, :].fill_(-self.bound) # Bounds of the function
        self.fun.bounds[1, :].fill_(self.bound)
        self.lb, self.ub = self.fun.bounds

        self.X_ei = self.get_initial_points(self.dim, self.n_init)
        self.Y_ei = torch.tensor([self.eval_objective(x) for x in self.X_ei], dtype=self.dtype, device=self.device).unsqueeze(-1)

    def eval_objective(self, x):
        """This is a helper function to unnormalize and evaluate a point."""
        return self.fun(unnormalize(x, self.fun.bounds))
    
    def get_initial_points(self, dim, n_pts):
        """Get the intial points using Sobol"""
        sobol = SobolEngine(dimension=dim, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init
    
    def get_data(self):
        """Get the generated BO data"""
        unnormalized_X = unnormalize(self.X_ei, self.fun.bounds)
        data = torch.cat((unnormalized_X, self.Y_ei), axis=1)
        return data
    
    def format(self, data, dim, n_init, batch_size, epochs):
        """Format the BO data into a pandas DataFrame"""
        num_rows = n_init + batch_size * epochs
        types = [0] * n_init + [i // batch_size + 1 for i in range(num_rows - n_init)]
        columns = ['Batch'] + [f'x{i+1}' for i in range(dim)] + ['y']
        types_tensor = torch.tensor(types, dtype=torch.int32).reshape(-1, 1)
        full_data = torch.cat((types_tensor, data), axis=1)
        df = pd.DataFrame(full_data.numpy(), columns=columns)
        return df


    def select_acquisition_function(self, model, best_f):
        """Select the acquisition function"""
        if self.acqf_type == 'qEI':
            return qExpectedImprovement(model, best_f)
        elif self.acqf_type == 'qUCB':
            return qUpperConfidenceBound(model, beta=0.1)
        elif self.acqf_type == 'qPI':
            return qProbabilityOfImprovement(model, best_f)
        else:
            raise ValueError(f"Unknown acquisition function type: {self.acqf_type}")

    def run(self):
        """Run the BO algo on the function"""
        for _ in range(self.epochs):

            train_Y = (self.Y_ei - self.Y_ei.mean()) / self.Y_ei.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-6))  # Noise constraints (1e-8, 1e-3)
            model = SingleTaskGP(self.X_ei, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Select acquisition function
            best_f = train_Y.max()
            acqf = self.select_acquisition_function(model, best_f)

            # Create a batch
            candidate, acq_value = optimize_acqf(
                acqf,
                bounds=torch.stack(
                    [
                        torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                        torch.ones(self.dim, dtype=self.dtype, device=self.device),
                    ]
                ),
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )

            Y_next = torch.tensor(
                [self.eval_objective(x) for x in candidate], dtype=self.dtype, device=self.device).unsqueeze(-1)

            # Append data
            self.X_ei = torch.cat((self.X_ei, candidate), axis=0)
            self.Y_ei = torch.cat((self.Y_ei, Y_next), axis=0)

            # Print current status
            print(f"{len(self.X_ei)}) Best value: {self.Y_ei.max().item():.2e}")

        best_idx = self.Y_ei.argmax().item()
        best_x = unnormalize(self.X_ei[best_idx], self.fun.bounds)
        # print(best_x, self.Y_ei.max().item())

        return best_x, self.Y_ei.max().item()


if __name__ == "__main__":

    # Ackley(dim=dim, negate=True).to(dtype=self.dtype, device=self.device)
    # qEI, qUCB, qPI
    optimizer = BayesianOptimization(fun=Ackley(dim=2, negate=True),batch_size=10, 
                                     dim=2, epochs=10, n_init=10, bound=32.768, acqf_type='qEI')
    x_max, y_max = optimizer.run()
    data = optimizer.get_data()
    print(data)
    print(x_max, y_max)
    full = optimizer.format(data, dim=2, n_init=10, batch_size=10, epochs=10)
    print(full)

    '''''
    optimizer = BayesianOptimization(fun=Levy(dim=2, negate=True),batch_size=10, 
                                     dim=2, epochs=5, n_init=10, bound=10)
    x_max, y_max = optimizer.run()
    data = optimizer.get_data()
    print(data)
    print(x_max, y_max)
    '''''
