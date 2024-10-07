import optuna
import copy
class ACOTuner:
    def __init__(self, aco_solver):
        self.aco_copy = copy.deepcopy(aco_solver)
        self.aco_orig = aco_solver

    def objective(self, trial):
        # Define the parameter space
        alpha = trial.suggest_float('alpha', 0.1, 2.0)
        beta = trial.suggest_float('beta', 3.0, 7.0)
        evaporation_rate = trial.suggest_float('evaporation_rate', 0.005, 0.5)
        deposit_factor = trial.suggest_float('deposit_factor', 0.01, 2)
        # n_ants = trial.suggest_int('n_ants', 50, 200)
        n_ranked_ants = trial.suggest_int('n_ranked_ants', 2, 20)


        self.aco_copy = copy.deepcopy(self.aco_orig)
        self.aco_copy.plot_obj = False
        self.aco_copy.silent = True
        self.aco_copy.set_tuning_params(alpha=alpha, beta=beta, evaporation_rate=evaporation_rate,
                                        deposit_factor=deposit_factor, n_ranked_ants=n_ranked_ants)

        _, best_obj = self.aco_copy.solve()
        return best_obj

    def tune_parameters(self, n_trials=100):
        self.study = optuna.create_study(direction='maximize')

        self.study.optimize(self.objective, n_trials=n_trials)

        return self.study.best_params

    def update_parameters(self):

        self.aco_orig.set_tuning_params(**self.study.best_params)
