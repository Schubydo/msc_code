import numpy as np
from bayesian_optimization import BayesianOptimization
from botorch.test_functions import Ackley
import matplotlib.pyplot as plt
import pandas as pd

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

def ci_bo(dim, acqf_type, batch_size=10, epochs=10, n_init=100, N_TRIALS=10, function=Ackley, lower_bound=-32.176, upper_bound=32.176, super_seed=1):

    fx_all = []
    for i in range(N_TRIALS):
        optimizer = BayesianOptimization(fun=function(dim=dim, negate=True), 
                                         batch_size=batch_size, 
                                         dim=dim, 
                                         epochs=epochs, 
                                         n_init=n_init, 
                                         lower_bound=lower_bound,
                                         upper_bound=upper_bound,
                                         seed=i+super_seed,
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


def compare_acquisition_functions(ucb, ucb_ci, ei, ei_ci, pi, pi_ci, filename='acquisition_function_comparisons.csv'):
    # Extract final max values and ci
    final_ucb = ucb[-1]
    final_ucb_ci = ucb_ci[-1]
    final_ei = ei[-1]
    final_ei_ci = ei_ci[-1]
    final_pi = pi[-1]
    final_pi_ci = pi_ci[-1]

    results = []
    for (acq1, final1, ci1), (acq2, final2, ci2) in [
        (('UCB', final_ucb, final_ucb_ci), ('EI', final_ei, final_ei_ci)),
        (('UCB', final_ucb, final_ucb_ci), ('PI', final_pi, final_pi_ci)),
        (('EI', final_ei, final_ei_ci), ('PI', final_pi, final_pi_ci)),
    ]:
        best_acq = acq1 if final1 > final2 else acq2
        superior = 'Yes' if final1 - ci1 > final2 + ci2 else 'No'
        results.append({
            'Comparison': f'{acq1} vs {acq2}',
            'Best': best_acq,
            'Superior': superior
        })

    # Create DataFrame for the results
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(filename, index=False)

    print(results_df)

def plot_acquisition_functions(ucb_2, ucb_ci_2, ei_2, ei_ci_2, pi_2, pi_ci_2):

    plt.figure(figsize=(10, 6))

    # Plot qUCB
    plt.plot(ucb_2, label='qUCB')
    plt.fill_between(np.arange(len(ucb_2)), ucb_2 - ucb_ci_2, ucb_2 + ucb_ci_2, alpha=0.2, label='95% CI (qUCB)')

    # Plot qEI
    plt.plot(ei_2, label='qEI')
    plt.fill_between(np.arange(len(ei_2)), ei_2 - ei_ci_2, ei_2 + ei_ci_2, alpha=0.2, label='95% CI (qEI)')

    # Plot qPI
    plt.plot(pi_2, label='qPI')
    plt.fill_between(np.arange(len(pi_2)), pi_2 - pi_ci_2, pi_2 + pi_ci_2, alpha=0.2, label='95% CI (qPI)')

    # Plot Optimum
    plt.plot([0, len(ucb_2)], [0, 0], "k--", lw=3, label='Optimum')

    plt.ylim([-20, 1])
    plt.title('Acquisition functions on 2D Ackley')
    plt.xlabel('Iteration')
    plt.ylabel('Max y value')
    plt.legend()
    plt.grid(True)
    plt.show()