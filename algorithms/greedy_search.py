from random import uniform
import numpy as np
# from core.environment.env import DroneSwarmSearch
from DSSE import DroneSwarmSearch, Actions


# __ desempenho __
# % de vezes que encontrou a pessoa.
# qnt media de acoes até achar.


def get_new_position(position: tuple, action: int) -> tuple:
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


def choose_drone_action(drone_position: tuple, greatest_prob_position) -> int:
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

def drones_colide(drones_positions: dict, new_drone_position: tuple) -> bool:
    return new_drone_position in drones_positions.values()


def policy(obs, agents):
    """
    Greedy approach: Rush and search for the greatest prob. point
    """
    drone_actions = {}
    prob_matrix = obs["drone0"]["observation"][1]
    n_drones = len(agents)

    drones_positions = {drone: obs[drone]["observation"][0] for drone in agents}
    # Get n_drones greatest probabilities.
    greatest_probs = np.argsort(prob_matrix, axis=None)[-n_drones:]
    
    for index, drone in enumerate(agents):
        greatest_prob = np.unravel_index(greatest_probs[index], prob_matrix.shape)

        drone_obs = obs[drone]["observation"]
        drone_action = choose_drone_action(drone_obs[0], greatest_prob)
        
        new_position = get_new_position(drone_obs[0], drone_action)
        
        # Avoid colision waiting 1 timeste
        if drones_colide(drones_positions, new_position):
            drone_actions[drone] = Actions.SEARCH.value
        else:
            drone_actions[drone] = drone_action
            drones_positions[drone] = new_position
    return drone_actions


def get_random_vector():
    return [uniform(-0.1, 0.1), uniform(-0.1, 0.1)]


if __name__ == "__main__":
    random_vector = get_random_vector()
    env = DroneSwarmSearch(
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        n_drones=4,  # range [1, 4]
        vector=random_vector,  # Aleatorio
        person_initial_position=[10, 10],  # Aleatorio
        disperse_constant=5,  # range [0, 10]
    )

    observations = env.reset()
    rewards = 0
    done = False
    while not done:
        actions = policy(observations, env.get_agents())
        observations, reward, _, done, info = env.step(actions)
        rewards += reward["total_reward"]
        done = any(done.values())
