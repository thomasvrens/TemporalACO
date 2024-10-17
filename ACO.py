import multiprocessing
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool
from TemporalACO.TaskNodes import *

class TeACO:
    def __init__(self, task_nodes, feasible_tasks,
                 task_energy, moving_energy, energy_to_depot,
                 aco_params, tug_props, max_tugs,
                 parallel=False, plot_obj=True, silent=False):

        self.validate_inputs(task_nodes, feasible_tasks,
                 task_energy, moving_energy, energy_to_depot,
                 aco_params, tug_props, max_tugs)

        self.tug_props = tug_props
        self.n_ants = aco_params['n_ants']
        self.n_iterations = aco_params['n_iterations']
        self.alpha = aco_params['alpha']
        self.beta = aco_params['beta']
        self.evaporation_rate = aco_params['evaporation_rate']
        self.deposit_factor = aco_params['deposit_factor']
        self.pheromone_scheme = aco_params['pheromone_scheme']
        self.n_ranked_ants = aco_params['n_ranked_ants']
        self.reset_every = aco_params['reset_every']
        self.max_tugs = max_tugs

        self.parallel = parallel
        self.plot_obj = plot_obj
        self.silent = silent

        # Init task node list
        self.task_nodes = task_nodes
        self.time_feasible_tasks = feasible_tasks
        self.task_energy = task_energy
        self.moving_energy = moving_energy
        self.energy_to_depot = energy_to_depot

        self.tow_indices = np.array([node.index for node in self.task_nodes if node.task_type=='towing'])
        self.charge_indices = np.array([node.index for node in self.task_nodes if node.task_type=='charging'])
        self.start_index = 0

        self.energy_task_and_return = self.moving_energy + self.task_energy + self.energy_to_depot
        self.task_objectives = np.array([node.task_objective for node in self.task_nodes])

        # Initialize pheromones and heuristics
        print("Initializing pheromones and heuristics...", end=" ")
        self.pheromones = self.initialize_pheromone()
        self.heuristics = self.initialize_heuristics()
        print("DONE")


        print(f"Initialized ACO solver module with alpha={self.alpha}, beta={self.beta}, evaporation_rate={self.evaporation_rate}")

    def set_tuning_params(self, **kwargs):
        valid_params = ['alpha', 'beta', 'evaporation_rate', 'n_ants', 'n_ranked_ants', 'deposit_factor']

        params_set = {key: value for key, value in kwargs.items() if key in valid_params and value is not None}
        invalid_params = [key for key in kwargs if key not in valid_params]

        if invalid_params:
            warnings.warn(f"Invalid parameters passed to set_tuning_params(): {', '.join(invalid_params)}", UserWarning)

        if not params_set:
            raise ValueError("No valid parameters were passed to set_tuning_params()")

        for key, value in params_set.items():
            setattr(self, key, value)

        if not self.silent:
            print("Tuning values set: " + ", ".join(f"{key} = {value}" for key, value in params_set.items()))

    def solve(self):
        if not self.silent:
            print(f"Starting ACO solver with {self.n_iterations} iterations and {self.n_ants} ants...")
        best_solution = None
        best_obj = 0

        it_best_objectives = []
        for i in tqdm(range(self.n_iterations)):
            if not self.parallel:
                ant_solutions = self.construct_ant_solutions()
            else:
                ant_solutions = self.construct_parallel_ant_solutions()

            # Calculate objective values
            objectives = [self.calc_solution_obj(solution) for solution in ant_solutions]
            it_best_objective = max(objectives)

            # Update pheromones
            self.update_pheromone(ant_solutions, objectives, i)

            it_best_index = objectives.index(it_best_objective)
            iteration_best = ant_solutions[it_best_index]
            it_best_objectives.append(it_best_objective)
            if it_best_objective > best_obj:
                best_solution = iteration_best
                best_obj = it_best_objective

        if self.plot_obj:
            plt.plot(it_best_objectives)
            plt.show()
        return best_solution, best_obj

    @classmethod
    def construct_single_ant_solution(cls, solver):
        tow_tasks = set(solver.tow_indices)
        solution = []

        while len(solution) < solver.max_tugs:
            last_task_ind = 0
            tug_route = [0]
            tug_battery = solver.tug_props['battery_cap']

            while True:
                tug_feasible_tasks_ind = solver.get_tug_feasible_tasks(last_task_ind, tow_tasks, tug_battery)
                if tug_feasible_tasks_ind is None or tug_feasible_tasks_ind.size == 0:
                    break

                next_task = solver.select_next_task(last_task_ind, tug_feasible_tasks_ind, tug_battery)

                # Update tug battery
                tug_battery += (solver.moving_energy[last_task_ind, next_task] + solver.task_energy[next_task])
                tug_battery = min(tug_battery, solver.tug_props['battery_cap'])

                # Append task to route and remove from available tasks
                tug_route.append(next_task)
                last_task_ind = tug_route[-1]
                if type(solver.task_nodes[next_task]) is TowNode and solver.task_nodes[next_task].alt_indices:
                    for ind in solver.task_nodes[next_task].alt_indices:
                        tow_tasks.remove(ind)
                elif type(solver.task_nodes[next_task]) is TowNode:
                    tow_tasks.remove(next_task)

            solution.append(tug_route)
        return solution

    def construct_ant_solutions(self):
        ant_solutions = []

        for k in range(self.n_ants):
            solution = self.construct_single_ant_solution(self)
            ant_solutions.append(solution)

        return ant_solutions

    def construct_parallel_ant_solutions(self):
        num_cores = multiprocessing.cpu_count()
        with Pool(processes=num_cores) as pool:
            ant_solutions = pool.map(self.construct_single_ant_solution, [self]*self.n_ants)
        return ant_solutions

    def calc_solution_obj(self, solution):
        solution_obj = 0
        for tug_path in solution:
            solution_obj += np.sum(self.task_objectives[np.array(tug_path)])

        return solution_obj

    def get_tug_feasible_tasks(self, last_task, available_set, tug_battery):
        time_feas_tasks = self.time_feasible_tasks[last_task]
        if len(time_feas_tasks) == 0:
            return None
        last_feas_task = time_feas_tasks[-1]

        # available_set = set(available_tasks)
        time_feas_set = set(time_feas_tasks)

        time_available_set = available_set & time_feas_set
        if not time_available_set:
            return None

        # if tug_battery >= self.tug_props['battery_cap']:
        #     return np.array(list(time_available_set))

        # Add charge task back if battery is not full
        if type(self.task_nodes[last_feas_task]) is ChargeNode and tug_battery < self.tug_props['battery_cap']:
            time_available_set.add(last_feas_task)

        time_available_tasks = np.array(list(time_available_set))
        task_and_return_energy = self.energy_task_and_return[last_task, time_available_tasks]
        remaining_energy = tug_battery + task_and_return_energy
        feas_task_ind = np.where(remaining_energy > 0)[0]
        feas_tasks = time_available_tasks[feas_task_ind]

        return feas_tasks

    def select_next_task(self, last_task, feas_tasks, tug_battery):

        if len(feas_tasks) == 1:
            return feas_tasks[0]

        pheromones = self.pheromones[last_task, feas_tasks]
        heuristics = self.heuristics[last_task, feas_tasks].copy()

        if type(self.task_nodes[feas_tasks[-1]]) is ChargeNode:
            heuristics[-1] = 1 - tug_battery / self.tug_props['battery_cap']

        p_values = pheromones ** self.alpha * heuristics ** self.beta
        probabilities = p_values / np.sum(p_values)

        # selected_index = np.random.choice(len(feas_tasks), p=probabilities)
        # Faster way to select index
        cumprob = np.cumsum(probabilities)
        rand_value = np.random.random()

        # Find the index corresponding to the random value in the cumulative probability
        selected_index = np.searchsorted(cumprob, rand_value)

        return feas_tasks[selected_index]

    def initialize_heuristics(self):
        n_task_nodes = len(self.task_nodes)
        h_mat = np.zeros((n_task_nodes, n_task_nodes))

        for i in range(n_task_nodes):
            reachable_nodes = self.time_feasible_tasks[i]
            for j in reachable_nodes:
                time_till_available = self.task_nodes[j].t_end - self.task_nodes[i].t_end
                assert time_till_available > 0, "Time till available should be positive between feasible tasks"
                objective = max(self.task_objectives[j], 0)
                h_value = objective / time_till_available
                h_mat[i, j] = h_value

        # Row-wise scaling between 0 and 1
        row_min = h_mat.min(axis=1, keepdims=True)
        row_max = h_mat.max(axis=1, keepdims=True)
        denominator = np.where(row_max != row_min, row_max - row_min, 1)

        h_norm = (h_mat - row_min) / denominator
        h_norm = np.nan_to_num(h_norm, nan=0)

        return h_norm

    def initialize_pheromone(self):
        return np.ones((len(self.task_nodes), len(self.task_nodes)))

    def update_pheromone(self, ant_solutions, objectives, iteration):
        # Evaporation
        self.pheromones *= (1 - self.evaporation_rate)

        # Sort objectives and solutions
        sorted_pairs = sorted(zip(objectives, ant_solutions), reverse=True)
        sorted_obj, sorted_solutions = map(list, zip(*sorted_pairs))
        max_obj = sorted_obj[0]
        min_obj = sorted_obj[-1]

        if type(self.reset_every) is int and iteration % self.reset_every == 0:
            self.pheromones = self.initialize_pheromone()

        if self.pheromone_scheme == 'ranked':
            for r in range(min(self.n_ranked_ants, len(ant_solutions))):
                pheromone_deposit = self.calc_pheromone_deposit(sorted_obj[r], min_obj, max_obj) * (self.n_ranked_ants - r)

                for tug_route in sorted_solutions[r]:
                    for i in range(len(tug_route)-1):
                        self.pheromones[tug_route[i], tug_route[i+1]] += pheromone_deposit
        else:
            for r in range(len(ant_solutions)):
                pheromone_deposit = self.calc_pheromone_deposit(objectives[r], min_obj, max_obj)

                for tug_route in ant_solutions[r]:
                    for i in range(len(tug_route)-1):
                        self.pheromones[tug_route[i], tug_route[i+1]] += pheromone_deposit

    def calc_pheromone_deposit(self, objective, min, max):
        scaled_obj = (objective - min) / (max - min)
        deposit = scaled_obj * self.deposit_factor
        return deposit

    def validate_inputs(self, task_nodes, feasible_tasks,
                 task_energy, moving_energy, energy_to_depot,
                 aco_params, tug_props, max_tugs):

        # Check task nodes starts with start node
        if not type(task_nodes[0]) is StartNode:
            raise TypeError("First TaskNode must be of type StartNode")

        # Only one StartNode
        if sum(1 for node in task_nodes if type(node) is StartNode) > 1:
            raise ValueError("task_nodes should only contain 1 StartNode")


        # Feasible tasks contain at most one charge index
        # Each feasible task should have moving energy associated with it
        # Moving energy should be negative
        for i, tasks in enumerate(feasible_tasks):
            for otask_index in tasks:
                num_charge = 0
                if type(task_nodes[otask_index]) is ChargeNode:
                    num_charge += 1

                if num_charge > 1:
                    raise ValueError(f"Feasible tasks at index {i} should only contain one ChargeNode")
                if num_charge == 1 and type(task_nodes[tasks[-1]]) is not ChargeNode:
                    raise TypeError(f"Last index of feasible tasks for task {i} should be the charge task")
                if np.isnan(moving_energy[i, otask_index]):
                    raise TypeError(f"Moving energy should not be nan at index {i}, {otask_index}")
                if moving_energy[i, otask_index] > 0:
                    raise ValueError(f"Moving energy should be negative at index {i}, {otask_index}")


        # Charge nodes positive energy, tow nodes negative
        for i, energy in enumerate(task_energy):
            if type(task_nodes[i]) is TowNode and energy > 0:
                raise ValueError(f"Task {i} is a tow task and should have negative energy")
            if type(task_nodes[i]) is ChargeNode and energy < 0:
                raise ValueError(f"Task {i} is a charge task and should have positive energy")

        # Energy to depot should be negative
        for i, energy in enumerate(energy_to_depot):
            if energy > 0:
                raise ValueError(f"Energy to depot should be negative at index {i}")

        # Max tugs must be integer
        if not isinstance(max_tugs, int):
            raise TypeError("max_tugs must be an integer")

        # Max tugs must be positive
        if not max_tugs > 0:
            raise ValueError("max_tugs must be positive")



