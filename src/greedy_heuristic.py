import numpy as np
from DSSE import Actions
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper
from .utils.recorder import PygameRecord


class GreedyAgent:
    def __call__(self, obs, agents):
        """
        Greedy approach: Rush and search for the greatest prob.
        """
        drone_actions = {}
        prob_matrix = obs["drone0"][1]
        n_drones = len(agents)

        drones_positions = {drone: obs[drone][0] for drone in agents}
        # Get n_drones greatest probabilities.
        greatest_probs = np.argsort(prob_matrix, axis=None)[-n_drones:]

        for index, drone in enumerate(agents):
            greatest_prob = np.unravel_index(greatest_probs[index], prob_matrix.shape)

            drone_obs = obs[drone]
            drone_action = self.choose_drone_action(drone_obs[0], greatest_prob)

            new_position = self.get_new_position(drone_obs[0], drone_action)

            # Avoid colision by waiting 1 timestep
            if self.drones_colide(drones_positions, new_position):
                drone_actions[drone] = Actions.SEARCH.value
            else:
                drone_actions[drone] = drone_action
                drones_positions[drone] = new_position
        return drone_actions

    def get_new_position(self, position: tuple, action: int) -> tuple:
        match action:
            case Actions.LEFT.value:
                new_position = (position[0] - 1, position[1])
            case Actions.RIGHT.value:
                new_position = (position[0] + 1, position[1])
            case Actions.UP.value:
                new_position = (position[0], position[1] - 1)
            case Actions.DOWN.value:
                new_position = (position[0], position[1] + 1)
            case Actions.UP_LEFT.value:
                new_position = (position[0] - 1, position[1] - 1)
            case Actions.UP_RIGHT.value:
                new_position = (position[0] + 1, position[1] - 1)
            case Actions.DOWN_LEFT.value:
                new_position = (position[0] - 1, position[1] + 1)
            case Actions.DOWN_RIGHT.value:
                new_position = (position[0] + 1, position[1] + 1)
            case _:
                new_position = position
        return new_position

    def choose_drone_action(self, drone_position: tuple, greatest_prob_position) -> int:
        greatest_prob_y, greatest_prob_x = greatest_prob_position
        drone_x, drone_y = drone_position
        is_x_greater = greatest_prob_x > drone_x
        is_y_greater = greatest_prob_y > drone_y
        is_x_lesser = greatest_prob_x < drone_x
        is_y_lesser = greatest_prob_y < drone_y
        is_x_equal = greatest_prob_x == drone_x
        is_y_equal = greatest_prob_y == drone_y

        if is_x_equal and is_y_equal:
            return Actions.SEARCH.value

        if is_x_equal:
            return Actions.DOWN.value if is_y_greater else Actions.UP.value
        elif is_y_equal:
            return Actions.LEFT.value if is_x_lesser else Actions.RIGHT.value

        # Movimento na diagonal
        if is_x_greater and is_y_greater:
            return Actions.DOWN_RIGHT.value
        elif is_x_greater and is_y_lesser:
            return Actions.UP_RIGHT.value
        elif is_x_lesser and is_y_greater:
            return Actions.DOWN_LEFT.value
        elif is_x_lesser and is_y_lesser:
            return Actions.UP_LEFT.value

    def drones_colide(self, drones_positions: dict, new_drone_position: tuple) -> bool:
        return new_drone_position in drones_positions.values()

    def __repr__(self) -> str:
        return "greedy"


see = input("Want to see ?? (y/n): ")
see = see.lower() == "y"


def env_creator(_):
    env = DroneSwarmSearch(
        render_mode="human" if see else "ansi",
        render_grid=True,
        drone_amount=4,
        grid_size=40,
        dispersion_inc=0.1,
        person_initial_position=(20, 20),
    )
    positions = [
        (20, 0),
        (20, 39),
        (0, 20),
        (39, 20),
    ]
    env = RetainDronePosWrapper(env, positions)
    return env


greedy_agent = GreedyAgent()

env = env_creator(None)

if see:
    with PygameRecord("greedy.gif", 5) as recorder:
        obs, info = env.reset()
        while env.agents:
            actions = greedy_agent(obs, env.agents)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            recorder.add_frame()
    exit(1)

rewards = []
actions = []
founds = 0
N_EVALS = 5_000
for _ in range(N_EVALS):
    i = 0
    obs, info = env.reset()
    reward_sum = 0
    while env.agents:
        action = greedy_agent(obs, env.agents)
        obs, rw, term, trunc, info = env.step(action)
        reward_sum += sum(rw.values())
        i += 1
    rewards.append(reward_sum)
    actions.append(i)

    for _, v in info.items():
        if v["Found"]:
            founds += 1
            break
print("Average reward: ", sum(rewards) / N_EVALS)
print("Mean actions: ", sum(actions) / N_EVALS)
print("Found %: ", founds / N_EVALS)
