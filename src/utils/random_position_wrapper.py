from pettingzoo.utils.wrappers import BaseParallelWrapper
from DSSE import DroneSwarmSearch
import random


class RandomPositionWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the reset function to randomize the positions of the drones
    """

    def __init__(self, env: DroneSwarmSearch):
        super().__init__(env)
        self.possible_positions = [
            (x, y) for x in range(self.env.grid_size) for y in range(self.env.grid_size)
        ]

    def reset(self, **kwargs):
        opt = kwargs.get("options", {})
        # Generate random positions for the drones
        opt["drones_positions"] = random.sample(
            self.possible_positions, len(self.env.possible_agents)
        )
        kwargs["options"] = opt

        return self.env.reset(**kwargs)


if __name__ == "__main__":
    env = DroneSwarmSearch(
        grid_size=9,
        timestep_limit=10,
        render_mode="human",
        render_grid=True,
        drone_amount=4,
    )
    env = RandomPositionWrapper(env)
    for _ in range(5):
        env.reset()
        while env.agents:
            env.step({agent: env.action_space(agent).sample() for agent in env.agents})
    env.close()
