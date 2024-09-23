import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool

class TeACO:
    def __init__(self, task_nodes, feasible_tasks,
                 task_energy, moving_energy, energy_to_depot,
                 aco_params, tug_props, max_tugs,
                 parallel=False, plot_obj=True):
        self.tug_props = tug_props
        self.n_ants = aco_params['n_ants']
        self.n_iterations = aco_params['n_iterations']
        self.alpha = aco_params['alpha']
        self.beta = aco_params['beta']
        self.evaporation_rate = aco_params['evaporation_rate']
        self.deposit_factor = aco_params['deposit_factor']
        self.pheromone_scheme = aco_params['pheromone_scheme']
        self.n_ranked_ants = aco_params['n_ranked_ants']
        self.max_tugs = max_tugs
        self.parallel = parallel
        self.plot_obj = plot_obj

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

    def solve(self):
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
            self.update_pheromone(ant_solutions, objectives)

            it_best_index = objectives.index(it_best_objective)
            iteration_best = ant_solutions[it_best_index]
            it_best_objectives.append(it_best_objective)
            if it_best_objective > best_obj:
                best_solution = iteration_best
                best_obj = it_best_objective

        if self.plot_obj:
            plt.plot(it_best_objectives)
            plt.show()
        return best_solution

    @classmethod
    def construct_single_ant_solution(cls, solver):
        tow_tasks = list(solver.tow_indices.copy())
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
                if solver.task_nodes[next_task].task_type == 'towing':
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
        fuel_savings = 0
        for tug_path in solution:
            fuel_savings += np.sum(self.task_objectives[np.array(tug_path)])

        return fuel_savings

    def get_tug_feasible_tasks(self, last_task, available_tasks, tug_battery):
        available_tasks = set(available_tasks)
        time_feas_tasks = self.time_feasible_tasks[last_task]
        if available_tasks.isdisjoint(set(time_feas_tasks)):
            return None

        if tug_battery >= self.tug_props['battery_cap']:
            return np.array(list(set(time_feas_tasks) & available_tasks))

        time_available_tasks = np.array(list(set(time_feas_tasks) & (available_tasks | set(self.charge_indices))))

        task_and_return_energy = self.energy_task_and_return[last_task, time_available_tasks]
        feas_task_ind = np.where(tug_battery + task_and_return_energy > 0)[0]
        feas_tasks = time_available_tasks[feas_task_ind]

        return feas_tasks

    def select_next_task(self, last_task, feas_tasks, tug_battery):

        if len(feas_tasks) == 1:
            return feas_tasks[0]

        pheromones = self.pheromones[last_task, feas_tasks]
        heuristics = self.heuristics[last_task, feas_tasks]

        if feas_tasks[-1] in self.charge_indices:
            heuristics[-1] = 1 - tug_battery / self.tug_props['battery_cap']

        p_values = pheromones ** self.alpha * heuristics ** self.beta
        probabilities = p_values / np.sum(p_values)

        selected_index = np.random.choice(len(feas_tasks), p=probabilities)

        return feas_tasks[selected_index]

    def initialize_heuristics(self):
        n_task_nodes = len(self.task_nodes)
        h_mat = np.zeros((n_task_nodes, n_task_nodes))

        for i in range(n_task_nodes):
            reachable_nodes = self.time_feasible_tasks[i]
            for j in reachable_nodes:
                time_till_available = self.task_nodes[j].t_end - self.task_nodes[i].t_end
                fuel_savings = self.task_objectives[j]
                h_value = fuel_savings / time_till_available
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

    def update_pheromone(self, ant_solutions, objectives):
        # Evaporation
        self.pheromones *= (1 - self.evaporation_rate)

        if self.pheromone_scheme == 'ranked':
            sorted_solutions = [x for _, x in sorted(zip(objectives, ant_solutions), reverse=True)]

            for r in range(min(self.n_ranked_ants - 1, len(ant_solutions))):
                pheromone_deposit = self.calc_pheromone_deposit(sorted_solutions[r]) * (self.n_ranked_ants - r)
                for tug_route in sorted_solutions[r]:
                    for i in range(len(tug_route)-1):
                        self.pheromones[tug_route[i], tug_route[i+1]] += pheromone_deposit
        else:
            for solution in ant_solutions:
                pheromone_deposit = self.calc_pheromone_deposit(solution)
                for tug_route in solution:
                    for i in range(len(tug_route)-1):
                        self.pheromones[tug_route[i], tug_route[i+1]] += pheromone_deposit

    def calc_pheromone_deposit(self, solution):
        deposit = self.calc_solution_obj(solution) * self.deposit_factor

        return deposit
