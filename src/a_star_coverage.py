"""
Module to implement the A* algorithm for coverage path planning.
"""
import time
import argparse
import numpy as np
from DSSE import CoverageDroneSwarmSearch, Actions
from aigyminsper.search.graph import State

MOVEMENTS = {
    Actions.UP: (0, -1),
    Actions.DOWN: (0, 1),
    Actions.LEFT: (-1, 0),
    Actions.RIGHT: (1, 0),
    Actions.UP_LEFT: (-1, -1),
    Actions.UP_RIGHT: (1, -1),
    Actions.DOWN_LEFT: (-1, 1),
    Actions.DOWN_RIGHT: (1, 1),
}


class DroneState(State):
    def __init__(self, position: tuple, prob_matrix: np.ndarray, visited: set):
        self.position = position
        self.prob_matrix = prob_matrix
        self.visited = visited
        self.action = None

    def successors(self, allow_zeros: bool = False) -> list['DroneState']:
        successors = []
        x, y = self.position

        for action, (dx, dy) in MOVEMENTS.items():
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < len(self.prob_matrix) and 0 <= new_y < len(self.prob_matrix[0]):
                if allow_zeros or self.prob_matrix[new_x, new_y] > 0:
                    new_state = DroneState((new_x, new_y), self.prob_matrix, self.visited | {(new_x, new_y)})
                    new_state.set_action(action)
                    successors.append(new_state)

        return successors

    def cost(self, prob_weight: int | float = 10, distance_weight: int | float = 0.5, revisit_penalty_value: int | float = 30) -> int | float:
        x, y = self.position
        prob_value = self.prob_matrix[x, y]

        high_prob_indices = np.argwhere(self.prob_matrix > 0)
        number_of_high_prob = high_prob_indices.shape[0]

        if number_of_high_prob == 0:
            return float('inf')

        distances = np.abs(high_prob_indices[:, 0] - x) + np.abs(high_prob_indices[:, 1] - y)
        min_distance = np.min(distances)

        max_prob = np.max(self.prob_matrix)
        normalized_prob = prob_value / max_prob if max_prob > 0 else 0

        max_distance = len(self.prob_matrix) + len(self.prob_matrix[0])
        normalized_distance = min_distance / max_distance

        revisit_penalty = 0
        if (x, y) in self.visited:
            revisit_penalty = revisit_penalty_value * (len(self.visited) / (number_of_high_prob + 1))

        heuristic_value = (
            -(normalized_prob * prob_weight)
            + (normalized_distance * distance_weight)
            + revisit_penalty
        )

        return heuristic_value

    def description(self):
        return f"DroneState Position = {self.position}"
    
    def env(self):
        return self.position
    
    def is_goal(self):
        return False
    
    def set_action(self, action: Actions):
        self.action = action

    def get_action(self) -> Actions:
        return self.action if self.action else Actions.SEARCH
    
    def __eq__(self, other: 'DroneState') -> bool:
        position_eq = self.position == other.position
        visited_eq = self.visited == other.visited
        action_eq = self.action == other.action
        return position_eq and visited_eq and action_eq

def a_star(observations: dict, agents: list, prob_matrix: np.ndarray, visited: set) -> dict:
    actions = {}
    will_visit = []
    for i, agent in enumerate(agents):
        current_position = (observations[agent][0][0], observations[agent][0][1])

        drone_state = DroneState(current_position, prob_matrix, visited[i])

        successors = drone_state.successors()

        if not successors:
            successors = drone_state.successors(allow_zeros=True)

        if not successors:
            continue

        next_state = min(successors, key=lambda state: state.cost())
        while next_state.position in will_visit:
            successors.remove(next_state)
            next_state = min(successors, key=lambda state: state.cost())
        will_visit.append(next_state.position)
        actions[agent] = next_state.get_action().value
        visited[i].add(next_state.position)
        prob_matrix[current_position[0], current_position[1]] = 0

    return actions


def main(num_drones: int):
    env = CoverageDroneSwarmSearch(
        drone_amount=num_drones,
        render_mode="human",
        timestep_limit=200,
        prob_matrix_path="src/min_matrix.npy"
    )

    center = env.grid_size // 2
    positions = [(center, center)]

    for i in range(1, num_drones):
        offset = (i // 2) + 1
        x_offset = offset * (-1 if i % 2 == 0 else 1)
        y_offset = offset * (-1 if (i + 1) % 2 == 0 else 1)
        new_position = (center + x_offset, center + y_offset)
        positions.append(new_position)

    opt = {
        "drones_positions": positions
    }

    observations, info = env.reset(options=opt)


    visited = [set([opt["drones_positions"][i]]) for i in range(num_drones)]

    prob_matrix = env.probability_matrix.get_matrix()
    step = 0
    while env.agents:
        step += 1

        actions = a_star(observations, env.agents, prob_matrix, visited)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        time.sleep(1)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_drones", type=int, required=True)
    args = argparser.parse_args()
    main(num_drones=args.num_drones)
