"""Example of implementing and training with an intrinsic curiosity model (ICM).

This type of curiosity-based learning trains a simplified model of the environment
dynamics based on three networks:
1) Embedding observations into latent space ("feature" network).
2) Predicting the action, given two consecutive embedded observations
("inverse" network).
3) Predicting the next embedded obs, given an obs and action
("forward" network).

The less the ICM is able to predict the actually observed next feature vector,
given obs and action (through the forwards network), the larger the
"intrinsic reward", which will be added to the extrinsic reward of the agent.

Therefore, if a state transition was unexpected, the agent becomes
"curious" and will further explore this transition leading to better
exploration in sparse rewards environments.

For more details, see here:
[1] Curiosity-driven Exploration by Self-supervised Prediction
Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
https://arxiv.org/pdf/1705.05363.pdf

This example:
    - demonstrates how to write a custom RLModule, representing the ICM from the paper
    above. Note that this custom RLModule does not belong to any individual agent.
    - demonstrates how to write a custom (PPO) TorchLearner that a) adds the ICM to its
    MultiRLModule, b) trains the regular PPO Policy plus the ICM module, using the
    PPO parent loss and the ICM's RLModule's own loss function.

We use a FrozenLake (sparse reward) environment with a custom map size of 12x12 and a
hard time step limit of 22 to make it almost impossible for a non-curiosity based
learners to learn a good policy.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

Use the `--no-curiosity` flag to disable curiosity learning and force your policy
to be trained on the task w/o the use of intrinsic rewards. With this option, the
algorithm should NOT succeed.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.
"""

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllFlattenWrapper
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from icm_learners import (
    ICM_MODULE_ID,
    PPOTorchLearnerWithCuriosity,
    DQNTorchLearnerWithCuriosity,
)
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import (
    IntrinsicCuriosityModel,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

parser = add_rllib_example_script_args(
    default_iters=2000,
    default_timesteps=10000000,
    default_reward=0.9,
)
parser.set_defaults(enable_new_api_stack=True)

# TODO: Study callbacks on the RLlib documentation
# TODO: Study the parser and runner functions
# TODO: Change the rest of the parameters
# TODO: Alter the stop condition


def env_creator(_):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 2
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=200,
        drone_amount=N_AGENTS,
        prob_matrix_path="../../../data/mat_9.npy",
    )
    env = AllFlattenWrapper(env)
    grid_size = env.grid_size
    print("Grid size: ", grid_size)
    positions = [
        (0, grid_size // 2),
        (grid_size - 1, grid_size // 2),
    ]
    env = RetainDronePosWrapper(env, positions)
    return env


"""
def main(args):
    ray.init()

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(
            num_rollout_workers=6, rollout_fragment_length="auto", num_envs_per_worker=4
        )
        .training(
            train_batch_size=8192 * 3,
            lr=8e-6,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            vf_clip_param=100000,
            minibatch_size=300,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [512, 256],
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
        name="PPO_" + input("Exp name: "),
        # resume=True,
        stop={"timesteps_total": 40_000_000},
        checkpoint_freq=25,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
"""


ENV_NAME = "DsseCoverage"


def main(_):
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    register_env(ENV_NAME, lambda config: ParallelPettingZooEnv(env_creator(config)))

    if args.algo not in ["DQN", "PPO"]:
        raise ValueError(
            "Curiosity example only implemented for either DQN or PPO! See the "
        )
    args.algo = "PPO"
    base_config = (
        PPOConfig()
        .environment(ENV_NAME)
        .env_runners(
            num_envs_per_env_runner=5 if args.algo == "PPO" else 1,
            # env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .training(
            learner_config_dict={
                # Intrinsic reward coefficient.
                "intrinsic_reward_coeff": 0.05,
                # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
                # L(total ICM) = (
                #     `forward_loss_weight` * L(forward)
                #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
                # )
                "forward_loss_weight": 0.2,
            }
        )
        # TODO: Try to do this like in https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_independent_learning.py
        .multi_agent(
            policies={
                "default_policy": PolicySpec(),
            },
            policy_mapping_fn=(lambda aid, *args, **kwargs: "default_policy"),
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        model_config={
                            "vf_share_layers": True,
                            "actor_critic_encoder_config": {
                                "fcnet_hiddens": [512, 256],
                                "fcnet_activation": "relu",
                            },
                        }
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config={
                            "feature_dim": 288,
                            "feature_net_hiddens": (256, 256),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (256, 256),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (256, 256),
                            "forward_net_activation": "relu",
                        },
                    ),
                }
            ),
            # Use a different learning rate for training the ICM.
            algorithm_config_overrides_per_module={
                ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
            },
        )
        .training(
            num_epochs=6,
            # Plug in the correct Learner class.
            learner_class=PPOTorchLearnerWithCuriosity,
            train_batch_size_per_learner=8000,
            train_batch_size=8192 * 3,
            lr=8e-6,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            vf_clip_param=100000,
            minibatch_size=300,
            num_sgd_iter=10,
        )
    )

    # Set PPO-specific hyper-parameters.
    # if args.algo == "PPO":
    #     base_config.training(
    #         num_epochs=6,
    #         # Plug in the correct Learner class.
    #         learner_class=PPOTorchLearnerWithCuriosity,
    #         train_batch_size_per_learner=8000,
    #         train_batch_size=8192 * 3,
    #         lr=8e-6,
    #         gamma=0.9999999,
    #         lambda_=0.9,
    #         use_gae=True,
    #         entropy_coeff=0.01,
    #         vf_clip_param=100000,
    #         minibatch_size=300,
    #         num_sgd_iter=10,
    #     )
    # elif args.algo == "DQN":
    #     base_config.training(
    #         # Plug in the correct Learner class.
    #         learner_class=DQNTorchLearnerWithCuriosity,
    #         train_batch_size_per_learner=128,
    #         lr=0.00075,
    #         replay_buffer_config={
    #             "type": "PrioritizedEpisodeReplayBuffer",
    #             "capacity": 500000,
    #             "alpha": 0.6,
    #             "beta": 0.4,
    #         },
    #         # Epsilon exploration schedule for DQN.
    #         epsilon=[[0, 1.0], [500000, 0.05]],
    #         n_step=(3, 5),
    #         double_q=True,
    #         dueling=True,
    #     )

    stop = {
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 8000,  # TODO: Check graph for this value
        NUM_ENV_STEPS_SAMPLED_LIFETIME: 40_000_000,
    }

    # TODO:  Modify this script to be able to choose the storage path
    run_rllib_example_script_experiment(
        base_config,
        args,
        stop=stop,
    )


if __name__ == "__main__":
    main(None)
