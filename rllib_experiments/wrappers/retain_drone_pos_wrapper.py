from pettingzoo.utils.wrappers import BaseParallelWrapper
from drone_swarm_search.DSSE import DroneSwarmSearch


class RetainDronePosWrapper(BaseParallelWrapper):
    """
    Wrapper that modifies the reset function to retain the drone positions
    """
    def __init__(self, env: DroneSwarmSearch, drone_positions: list):
        super().__init__(env)
        if len(drone_positions) != len(self.env.possible_agents):
            raise ValueError("Drone positions must have the same length as the number of possible agents")
        self.drone_positions = drone_positions

    
    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        return obs, reward, terminated, truncated, infos
    
    def reset(self, **kwargs):
        options = {
            "drone_positions": self.drone_positions
        }
        obs, infos = self.env.reset(options=options)
        return obs, infos

