import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool

class TaskNode:
    def __init__(self, task_type, index, flight=None, t_end=None, starting_depot=None, charging_depot=None, t_start_charge=None, charge_interval=None, charge_index=None, task_objective=None):
        self.task_type = task_type
        if task_type == 'towing':
            self.task_id = 'TOW-' + flight['ac_id']
            self.ac_id = flight['ac_id']
            self.direction = flight['direction']
            self.t_start = flight['timestamp']
            self.t_end = t_end
            self.start_node = flight['start_node']
            self.end_node = flight['end_node']
            self.index = index
            self.tow_index = index - 1
            self.ac_class = flight['ac_class']
            self.task_objective = task_objective
        elif task_type == 'start':
            self.task_id = 'START'
            self.t_start = -2
            self.t_end = -1
            self.start_node = starting_depot
            self.end_node = starting_depot
            self.ac_class = "NA"
            self.task_objective = 0
        elif task_type == 'charging':
            self.task_id = 'CHARGE-' + str(charge_index)
            self.t_start = t_start_charge
            self.t_end = self.t_start + charge_interval - 0.00001
            self.start_node = charging_depot
            self.end_node = charging_depot
            self.charge_index = charge_index
            self.index = index
            self.ac_class = "NA"
            self.task_objective = 0
        else:
            raise ValueError(f"Unkown task type when creating task node: {task_type}")

    def print_node(self):
        if self.task_type == 'towing':
            print(f"Task id: {self.task_id}, start time: {self.t_start:.3f}, end time: {self.t_end:.3f}, start node: {self.start_node}, end node: {self.end_node}")
        else:
            print(f"Task id: {self.task_id}")

    def get_task_dict(self):
        task_dict = {
            'start_node': self.start_node,
            'end_node': self.end_node,
            'ac_class': self.ac_class
        }
        return task_dict

class ACOSolver:
    def __init__(self, layout, flight_schedule, tug_props, aco_params, max_tugs, parallel=False):
        self.layout = layout
        self.flight_schedule = flight_schedule
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

        # Init task node list
        print("Initializing task nodes and time feasible tasks...", end=" ")
        self.task_nodes = self.generate_task_nodes()
        self.tow_indices = np.array([node.index for node in self.task_nodes if node.task_type=='towing'])
        self.charge_indices = np.array([node.index for node in self.task_nodes if node.task_type=='charging'])
        self.start_index = 0
        self.task_ids = []
        self.task_dict = {}
        for task in self.task_nodes:
            self.task_dict[task.task_id] = task
            self.task_ids.append(task.task_id)

        # Task start and end times
        self.task_start_times = np.array([node.t_start for node in self.task_nodes])
        self.task_end_times = np.array([node.t_end for node in self.task_nodes])

        # Time feasible task list
        self.time_feasible_tasks  = self.compute_feasible_tasks()
        print("DONE")

        # Precompute battery discharges
        print("Precomputing energy matrices and task fuel savings...", end=" ")
        self.task_energy = np.array([-self.layout.task_energy(node.get_task_dict()) for node in self.task_nodes])
        self.task_energy[self.charge_indices] = self.tug_props['charge_power'] * self.tug_props['charge_interval'] / 3600
        self.moving_energy = np.array([[-self.layout.free_moving_energy(i.end_node, j.start_node) for j in self.task_nodes] for i in self.task_nodes])
        charging_depot = self.layout.depots[0]
        self.energy_to_depot = np.array([-self.layout.free_moving_energy(node.end_node, charging_depot) for node in self.task_nodes])
        self.energy_task_and_return = self.moving_energy + self.task_energy + self.energy_to_depot
        self.task_objectives = np.array([node.task_objective for node in self.task_nodes])
        print("DONE")


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

    def generate_task_nodes(self):
        task_nodes = [TaskNode('start', 0, starting_depot=self.layout.depots[0])]

        for flight in self.flight_schedule:
            t_end = flight['timestamp'] + self.layout.task_time(flight)
            task_fuel = self.layout.task_fuel(flight)
            task = TaskNode('towing', len(task_nodes), flight=flight, t_end=t_end, task_objective=task_fuel)
            task_nodes.append(task)

        t_first_flight = task_nodes[1].t_start
        t_last_flight = task_nodes[-1].t_start
        t = t_first_flight
        while t <= t_last_flight:
            task = TaskNode('charging', len(task_nodes),
                            charging_depot=self.layout.depots[0],
                            t_start_charge=t,
                            charge_index=len(task_nodes)-len(self.flight_schedule)-1,
                            charge_interval=self.tug_props['charge_interval'])
            task_nodes.append(task)
            t += self.tug_props['charge_interval']

        return task_nodes

    def compute_feasible_tasks(self):
        feasible_tasks = []

        for task in self.task_nodes:
            feas_tasks = []
            if task.task_id == 'START':
                feas_tasks = self.tow_indices
            else:
                for other_task in self.task_nodes:
                    if task.t_end + self.layout.free_moving_time(task.end_node, other_task.start_node) < other_task.t_start\
                            and other_task.t_start - task.t_end < self.tug_props['max_idle_time']:
                        feas_tasks.append(other_task.index)
                        # Only add the next available charging task
                        if other_task.task_type == 'charging':
                            break

            feasible_tasks.append(np.array(feas_tasks))

        return feasible_tasks

    def get_feasible_tasks(self, from_task):
        return self.time_feasible_tasks[from_task]

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

    def extract_aco_solution(self, solution):
        tug_schedule = {}

        for task_chain in solution:
            tug_id = 'GV-' + str(len(tug_schedule))
            tug_schedule[tug_id] = []
            for task in task_chain:
                task_node = self.task_nodes[task]
                if task_node.task_type == 'towing':
                    task_dict = {
                        'ac_id': task_node.ac_id,
                        'direction': task_node.direction,
                        't_start': task_node.t_start,
                        't_end': task_node.t_end,
                        'start_node': task_node.start_node,
                        'end_node': task_node.end_node,
                        'ac_class': task_node.ac_class
                    }
                    tug_schedule[tug_id].append(task_dict)
                elif task_node.task_type == 'charging':
                    task_dict = {
                        'ac_id': task_node.task_id,
                        'direction': "CH",
                        't_start': task_node.t_start,
                        't_end': task_node.t_end,
                        'start_node': task_node.start_node,
                        'end_node': task_node.end_node
                    }
                    tug_schedule[tug_id].append(task_dict)

        return tug_schedule