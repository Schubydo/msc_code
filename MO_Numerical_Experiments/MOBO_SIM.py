import os
import torch
import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list


class BayesianOptimization:
    def __init__(self, BATCH_SIZE = 4, N_BATCH = 20, INITIAL_SAMPLES=6):
        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.SMOKE_TEST = os.environ.get("SMOKE_TEST")
        self.problem = BraninCurrin(negate=True).to(**self.tkwargs)
        self.NOISE_SE = torch.tensor([15.19, 0.63], **self.tkwargs)
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_RESTARTS = 10 if not self.SMOKE_TEST else 2
        self.RAW_SAMPLES = 512 if not self.SMOKE_TEST else 4
        self.standard_bounds = torch.zeros(2, self.problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.N_BATCH = N_BATCH if not self.SMOKE_TEST else 5
        self.MC_SAMPLES = 128 if not self.SMOKE_TEST else 16
        self.verbose = True
        self.hvs_qparego, self.hvs_qehvi, self.hvs_qnehvi, self.hvs_random = [], [], [], []
        self.INITIAL_SAMPLES = INITIAL_SAMPLES
        
        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def generate_initial_data(self):
        train_x = draw_sobol_samples(bounds=self.problem.bounds, n=self.INITIAL_SAMPLES, q=1).squeeze(1)
        train_obj_true = self.problem(train_x)
        train_obj = train_obj_true + torch.randn_like(train_obj_true) * self.NOISE_SE
        return train_x, train_obj, train_obj_true

    def initialize_model(self, train_x, train_obj):
        train_x = normalize(train_x, self.problem.bounds)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            train_yvar = torch.full_like(train_y, self.NOISE_SE[i] ** 2)
            models.append(
                FixedNoiseGP(
                    train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qehvi_and_get_observation(self, model, train_x, train_obj, sampler):
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, self.problem.bounds)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.problem.ref_point,
            Y=pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        new_obj_true = self.problem(new_x)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * self.NOISE_SE
        return new_x, new_obj, new_obj_true

    def optimize_qnehvi_and_get_observation(self, model, train_x, train_obj, sampler):
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.ref_point.tolist(),
            X_baseline=normalize(train_x, self.problem.bounds),
            prune_baseline=True,
            sampler=sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        new_obj_true = self.problem(new_x)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * self.NOISE_SE
        return new_x, new_obj, new_obj_true

    def optimize_qnparego_and_get_observation(self, model, train_x, train_obj, sampler):
        train_x = normalize(train_x, self.problem.bounds)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(self.problem.num_objectives, **self.tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.standard_bounds,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_x = unnormalize(candidates.detach(), bounds=self.problem.bounds)
        new_obj_true = self.problem(new_x)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * self.NOISE_SE
        return new_x, new_obj, new_obj_true

    def run_optimization(self):
        train_x_qparego, train_obj_qparego, train_obj_true_qparego = self.generate_initial_data(n=2 * (self.problem.dim + 1))
        mll_qparego, model_qparego = self.initialize_model(train_x_qparego, train_obj_qparego)
        
        train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        train_x_random, train_obj_random, train_obj_true_random = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        
        mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi)
        mll_qnehvi, model_qnehvi = self.initialize_model(train_x_qnehvi, train_obj_qnehvi)

        bd = DominatedPartitioning(ref_point=self.problem.ref_point, Y=train_obj_true_qparego)
        volume = bd.compute_hypervolume().item()
        self.hvs_qparego.append(volume)
        self.hvs_qehvi.append(volume)
        self.hvs_qnehvi.append(volume)
        self.hvs_random.append(volume)

        for iteration in range(1, self.N_BATCH + 1):
            t0 = time.monotonic()
            
            fit_gpytorch_mll(mll_qparego)
            fit_gpytorch_mll(mll_qehvi)
            fit_gpytorch_mll(mll_qnehvi)

            qparego_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES]))
            qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES]))
            qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES]))

            new_x_qparego, new_obj_qparego, new_obj_true_qparego = self.optimize_qnparego_and_get_observation(
                model_qparego, train_x_qparego, train_obj_qparego, qparego_sampler
            )
            new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = self.optimize_qehvi_and_get_observation(
                model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler
            )
            new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi = self.optimize_qnehvi_and_get_observation(
                model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
            )
            new_x_random, new_obj_random, new_obj_true_random = self.generate_initial_data(n=self.BATCH_SIZE)

            train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
            train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
            train_obj_true_qparego = torch.cat([train_obj_true_qparego, new_obj_true_qparego])

            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
            train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])

            train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

            train_x_random = torch.cat([train_x_random, new_x_random])
            train_obj_random = torch.cat([train_obj_random, new_obj_random])
            train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random])

            for hvs_list, train_obj in zip(
                (self.hvs_random, self.hvs_qparego, self.hvs_qehvi, self.hvs_qnehvi),
                (train_obj_true_random, train_obj_true_qparego, train_obj_true_qehvi, train_obj_true_qnehvi),
            ):
                bd = DominatedPartitioning(ref_point=self.problem.ref_point, Y=train_obj)
                volume = bd.compute_hypervolume().item()
                hvs_list.append(volume)

            mll_qparego, model_qparego = self.initialize_model(train_x_qparego, train_obj_qparego)
            mll_qehvi, model_qehvi = self.initialize_model(train_x_qehvi, train_obj_qehvi)
            mll_qnehvi, model_qnehvi = self.initialize_model(train_x_qnehvi, train_obj_qnehvi)

            t1 = time.monotonic()

            if self.verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI) = "
                    f"({self.hvs_random[-1]:>4.2f}, {self.hvs_qparego[-1]:>4.2f}, {self.hvs_qehvi[-1]:>4.2f}, {self.hvs_qnehvi[-1]:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")
        return self.hvs_random, self.hvs_qparego, self.hvs_qehvi, self.hvs_qnehvi
            

if __name__ == "__main__":
    # Instantiate the Bayesian Optimization class
    bo = BayesianOptimization(BATCH_SIZE= 4,N_BATCH= 10,INITIAL_SAMPLES= 6)

    # Run the optimization process
    random, qparego, qehvi, qnehvi = bo.run_optimization()