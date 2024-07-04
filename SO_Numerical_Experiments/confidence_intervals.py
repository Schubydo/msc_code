import numpy as np
from bayesian_optimization import BayesianOptimization
from botorch.test_functions import Ackley
import matplotlib.pyplot as plt

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

def ci_bo(dim, acqf_type, batch_size=10, epochs=10, n_init=100, N_TRIALS=10, function=Ackley, lower_bound=-32.176, upper_bound=32.176):

    fx_all = []
    for i in range(N_TRIALS):
        optimizer = BayesianOptimization(fun=function(dim=dim, negate=True), 
                                         batch_size=batch_size, 
                                         dim=dim, 
                                         epochs=epochs, 
                                         n_init=n_init, 
                                         lower_bound=lower_bound,
                                         upper_bound=upper_bound,
                                         seed=i,
                                         acqf_type=acqf_type)
        
        optimizer.run()
        data = optimizer.get_data()
        data_np = data.numpy()
        y = data_np[:, -1]
        fx = np.maximum.accumulate(y)
        fx_all.append(fx)

    fx_all = np.array(fx_all)
    fx_mean = fx_all.mean(axis=0)
    fx_ci = ci(fx_all, N_TRIALS)

    return fx_mean, fx_ci