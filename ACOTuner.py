import optuna
import copy
from datetime import datetime
import pickle
import os
import matplotlib.pyplot as plt

class ACOTuner:
    def __init__(self, aco_solver, tuner_params, fs_info):
        self.aco_copy = copy.deepcopy(aco_solver)
        self.aco_orig = aco_solver

        self.aco_copy.plot_obj = False
        self.aco_copy.silent = True

        self.n_iterations = tuner_params['n_iterations']
        self.n_ants = tuner_params['n_ants']

        self.n_trials = tuner_params['n_trials']
        self.runs_per_trial = tuner_params['runs_per_trial']

        self.alpha_range = (tuner_params['alpha_range'][0], tuner_params['alpha_range'][1])
        self.beta_range = (tuner_params['beta_range'][0], tuner_params['beta_range'][1])
        self.evap_range = (tuner_params['evap_range'][0], tuner_params['evap_range'][1])
        self.depos_range = (tuner_params['depos_range'][0], tuner_params['depos_range'][1])
        self.ranked_range = (tuner_params['ranked_range'][0], tuner_params['ranked_range'][1])

        self.tuner_params = tuner_params
        self.fs_info = fs_info

        if fs_info:
            print('\nFS Information:')
            for key, value in fs_info.items():
                print(f'{key}: {value}')

        print(f"\nTuner Run Information:\n"
              f"Number of Trials: {tuner_params['n_trials']}\n"
              f"Runs per Trial: {tuner_params['runs_per_trial']}\n"
              f"Number of Iterations: {tuner_params['n_iterations']}\n"
              f"Number of Ants: {tuner_params['n_ants']}\n"
              f"Initial Pheromone: {tuner_params['initial_pheromone']}\n"
              f"Pheromone Scheme: {tuner_params['pheromone_scheme']}\n\n"
              f"Alpha Range: {tuner_params['alpha_range']}\n"
              f"Beta Range: {tuner_params['beta_range']}\n"
              f"Evaporation Rate Range: {tuner_params['evap_range']}\n"
              f"Deposit Factor Range: {tuner_params['depos_range']}\n"
              f"Ranked Ants Range: {tuner_params['ranked_range']}")

    def single_run(self, plot_file=None):

        self.aco_copy.initialize_pheromone()
        _, best_obj = self.aco_copy.solve(plot_file=plot_file)
        return best_obj

    def multi_run(self):
        objectives = []

        for i in range(self.runs_per_trial):
            obj = self.single_run()
            objectives.append(obj)

        avg_obj = sum(objectives) / len(objectives)
        return avg_obj


    def objective(self, trial):
        # Define the parameter space
        alpha = trial.suggest_float('alpha', self.alpha_range[0], self.alpha_range[1])
        beta = trial.suggest_float('beta', self.beta_range[0], self.beta_range[1])
        evaporation_rate = trial.suggest_float('evaporation_rate', self.evap_range[0], self.evap_range[1])
        deposit_factor = trial.suggest_float('deposit_factor', self.depos_range[0], self.depos_range[1])
        n_ranked_ants = trial.suggest_int('n_ranked_ants', self.ranked_range[0], self.ranked_range[1])

        self.aco_copy.set_tuning_params(alpha=alpha, beta=beta, evaporation_rate=evaporation_rate,
                                        deposit_factor=deposit_factor, n_ranked_ants=n_ranked_ants)

        objective = self.multi_run()
        return objective

    def tune_parameters(self):
        self.study = optuna.create_study(direction='maximize')

        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.store_results()

        return self.study.best_params

    def update_parameters(self):

        self.aco_orig.set_tuning_params(**self.study.best_params)

    def store_results(self):
        print("Doing test runs and storing tuner run ...")

        store_dir = '../TA-Tester/opti_runs/'
        time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = f"optuna_run_{time_now}_iter{self.n_iterations}_ants{self.n_ants}_obj{self.study.best_value:.2f}"
        dir_path = store_dir + run_dir
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, 'best_params.pkl'), 'wb') as f:
            pickle.dump(self.study.best_params, f)

        with open(os.path.join(dir_path, 'study.pkl'), 'wb') as f:
            pickle.dump(self.study, f)

        # Do 5 test runs
        self.aco_copy.set_tuning_params(**self.study.best_params)
        self.aco_copy.silent = False
        test_objectives = []
        n_test_runs = 5
        for i in range(n_test_runs):
            plot_name = os.path.join(dir_path, f'test_run{i}.png')
            obj = self.single_run(plot_file=plot_name)
            test_objectives.append(obj)
        avg_test_obj = sum(test_objectives) / len(test_objectives)

        # Write to text
        with open(os.path.join(dir_path, 'run_summary.txt'), 'w') as f:
            if self.fs_info:
                f.write('FS INFORMATION:\n')
                for key, value in self.fs_info.items():
                    f.write(f'{key}: {value}\n')

            f.write("\nTUNER RUN SETTINGS\n")
            for key, value in self.tuner_params.items():
                f.write(f"{key}: {value}\n")

            f.write("\nOPTUNA RESULTS\n")
            f.write(f"Best Parameters:\n")
            for key, value in self.study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Objective Value: {self.study.best_value:.2f}\n")

            f.write(f"\n{n_test_runs} TEST RUNS\n")
            f.write("Objectives\n")
            for i in range(len(test_objectives)):
                f.write(f"{i}: {test_objectives[i]:.2f}\n")
            f.write(f"\nAverage: {avg_test_obj:.2f}\n")

        # Some optuna plots
        # Plot optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.gcf().savefig(os.path.join(dir_path, 'optimization_history.png'))
        plt.close()

        # Plot parallel coordinate
        optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        plt.gcf().savefig(os.path.join(dir_path, 'parallel_coordinate.png'))
        plt.close()

        # Plot parameter importances
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.gcf().savefig(os.path.join(dir_path, 'param_importances.png'))
        plt.close()

        # Plot contour plot
        optuna.visualization.matplotlib.plot_contour(self.study)
        plt.gcf().savefig(os.path.join(dir_path, 'contour_plot.png'))
        plt.close()

        print(f"\nTuner results stored in {dir_path}")



