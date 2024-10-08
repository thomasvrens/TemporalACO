
class TaskNode:
    def __init__(self, task_id, t_start, t_end, start_node, end_node, index, task_objective):
        self.task_id = task_id
        self.t_start = t_start
        self.t_end = t_end
        self.start_node = start_node
        self.end_node = end_node
        self.index = index
        self.task_objective = task_objective

        self.task_type = None




class TowNode(TaskNode):

    def __init__(self, task_id, t_start, t_end, start_node, end_node, index, task_objective, ac_id, tow_id, ac_cat):

        super().__init__(task_id, t_start, t_end, start_node, end_node, index, task_objective)

        self.ac_id = ac_id
        self.tow_id = tow_id
        self.alt_indices = None
        self.task_type = "towing"
        self.ac_cat = ac_cat

class ChargeNode(TaskNode):
    def __init__(self, task_id, t_start, depot_node, index, charge_interval, task_objective=0):

        start_node = depot_node
        end_node = depot_node
        t_end = t_start + charge_interval

        super().__init__(task_id, t_start, t_end, start_node, end_node, index, task_objective)

        self.task_type = "charging"

class StartNode(TaskNode):

    def __init__(self, task_id, starting_time, starting_node, index):

        start_node = starting_node
        end_node = start_node
        t_start = starting_time
        t_end = starting_time
        index = index
        super().__init__(task_id, t_start, t_end, start_node, end_node, index, 0)

        self.task_type = "start"
