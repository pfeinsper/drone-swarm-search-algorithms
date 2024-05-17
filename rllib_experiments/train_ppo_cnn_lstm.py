import os
import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from DSSE.environment.wrappers.communication_wrapper import CommunicationWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.tune.registry import register_env
from torch import nn
import torch


class CNNModel(TorchRNN, nn.Module):
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
        num_outputs = act_space.n
        nn.Module.__init__(self)
        super().__init__(obs_space, act_space, num_outputs, model_config, name, **kw)

        flatten_size = 32 * (obs_space[1].shape[0] - 7 - 3) * (obs_space[1].shape[0] - 7 - 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(8, 8),
                stride=(1, 1),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(1, 1),
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.linear = nn.Linear(obs_space[0].shape[0], 512)

        self.lstm_state_size = 256
        self.lstm = nn.LSTM(512, self.lstm_state_size, batch_first=True)

        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )
        print("NUM OUTPUTS: ", num_outputs)
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)
        # Holds the current "base" output (before logits layer).
        self._value_out = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h
    
    def value_function(self):
        return self._value_out.flatten()
    
    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        scalar_inputs = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)

        flat_inputs = scalar_inputs.flatten(start_dim=1)
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        lstm_out, new_state = self.forward_rnn(inputs, state, seq_lens)
        lstm_out = torch.reshape(lstm_out, [-1, self.lstm_state_size])

        value_input = torch.cat((cnn_out, lstm_out), dim=1)
        value_input = self.join(value_input)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), new_state
    
    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        linear_out = nn.functional.tanh(self.linear(inputs))

        lstm_out, [h, c] = self.lstm(linear_out, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])

        return lstm_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


def env_creator(args):
    """
    Petting Zoo environment for search of shipwrecked people.
    check it out at
        https://github.com/pfeinsper/drone-swarm-search
    or install with
        pip install DSSE
    """
    env = DroneSwarmSearch(
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
    env = AllPositionsWrapper(env)
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto")
        .training(
            train_batch_size=4096,
            lr=1e-5,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            sgd_minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "CNNModel",
                "use_lstm": False,
                "lstm_cell_size": 256,
                "_disable_preprocessor_api": True,
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "PPO",
        name="PPO_LSTM_M",
        resume=True,
        stop={"timesteps_total": 20_000_000, "episode_reward_mean": 1.82},
        checkpoint_freq=15,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
