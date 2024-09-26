import optuna
import copy
class ACOTuner:
    def __init__(self, aco_solver):
        self.aco_copy = copy.deepcopy(aco_solver)
        self.aco_orig = aco_solver

    def objective(self, trial):
        # Define the parameter space
        alpha = trial.suggest_float('alpha', 0.1, 5.0)
        beta = trial.suggest_float('beta', 0.1, 5.0)
        evaporation_rate = trial.suggest_float('evaporation_rate', 0.005, 0.5)


        self.aco_copy = copy.deepcopy(self.aco_orig)
        self.aco_copy.set_tuning_params(alpha=alpha, beta=beta, evaporation_rate=evaporation_rate)
        self.aco_copy.plot_obj = False

        _, best_obj = self.aco_copy.solve()
        return best_obj

    def tune_parameters(self, n_trials=100):
        self.study = optuna.create_study(direction='maximize')

        self.study.optimize(self.objective, n_trials=n_trials)

        return self.study.best_params