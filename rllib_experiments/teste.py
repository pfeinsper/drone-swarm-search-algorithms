import os
import pathlib
from drone_swarm_search.DSSE import DroneSwarmSearch
from wrappers import AllPositionsWrapper, RetainDronePosWrapper, TopNProbsWrapper
# from DSSE.environment.wrappers import AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
import torch
from torch import nn
import random


class MLPModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        # self.model = nn.Sequential(
        #     nn.Linear(obs_space.shape[0], 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),  # Adicionando dropout para regularização
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        # )
        
        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_ = input_dict["obs"].float()
        value_input = self.model(input_)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()

def ramdom_position(centro_x, centro_y, alcance, num_posicoes=4):
    # Gerar todas as posições possíveis dentro do alcance
    posicoes_possiveis = [(x, y) for x in range(centro_x - alcance, centro_x + alcance + 1)
                          for y in range(centro_y - alcance, centro_y + alcance + 1)]

    # Selecionar num_posicoes posições aleatoriamente das possíveis
    posicoes_aleatorias = random.sample(posicoes_possiveis, num_posicoes)

    return posicoes_aleatorias

def env_creator(args):
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=20,
        dispersion_inc=0.08,
        person_initial_position=(10, 10),
        person_amount=5,
        render_mode="ansi",
    )
    
    env = TopNProbsWrapper(env, 10)
    env = RetainDronePosWrapper(env, ramdom_position(10, 10, 3, 4))
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)
    
    replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 50000,
        "prioritized_replay_alpha": 0.8,
        "prioritized_replay_beta": 0.6,
        "prioritized_replay_eps": 3e-4,
    }
    
    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=8, rollout_fragment_length=128)
        .framework("torch")
        .resources(num_gpus=0)
        .training(
            lr=1e-4,
            # lr_schedule= [[timestep, value], [timestep, value],],
            tau=0.01,
            noisy=True,
            gamma=0.99995,
            train_batch_size=512,
            replay_buffer_config=replay_config,
            model={
                "custom_model": "MLPModel",
                "_disable_preprocessor_api": True,
            },
            target_network_update_freq=800,
            double_q=False,
        )
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "DQN",
        name="DSSE",
        stop={"timesteps_total": 5_000_000 if not os.environ.get("CI") else 50_000},
        checkpoint_freq=10,
        # local_dir=local_dir,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
    
# Finalize Ray to free up resources
ray.shutdown()
