import json

from stable_baselines3 import PPO

from hpc_rl_simulator.agent import CustomActorCriticPolicy, available_models
from hpc_rl_simulator.utils import init_dir_from_args, init_training_env
from hpc_rl_simulator.env import GymSchedulerEnv

if __name__ == '__main__':

    with open('ppo-conf.json', 'r') as f:
        config = json.load(f)

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(config)
    # create environment
    env = init_training_env(workload_file, GymSchedulerEnv, config)
    if config['trained_model']:
        model = PPO.load(config['trained_model'], env=env)
    else:
        if config['actor_model'] in available_models and config['critic_model'] in available_models:
            policy_args = {
                'attn': False,
                'actor_model': config['actor_model'],
                'critic_model': config['critic_model']
            }
        else:
            print(
                f"Invalid model name: {config['actor_model']} or {config['critic_model']} "
                f"not in available models: {', '.join(available_models)}")
            print(f"Using default model: kernel and critic_lg")
            policy_args = {}

        model = PPO(
            CustomActorCriticPolicy,
            env,
            learning_rate=config['lr'],
            n_steps=config['rollout_steps'],
            n_epochs=config['epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            normalize_advantage=False,
            target_kl=config['target_kl'],
            stats_window_size=config['stats_window_size'],
            policy_kwargs=policy_args,
            tensorboard_log=log_data_dir,
            verbose=config['verbose'],
            seed=config['seed'],
            device=config['device'],
        )
        env_steps = config['rollout_steps'] * config['num_rollouts']
        print(":AGENT-PPO: Learning")
        model.learn(total_timesteps=400000)
        model.save(f"{model_dir}ppo_HPC")
        print(f":AGENT-PPO: Trained model saved at: {model_dir}ppo_HPC")
