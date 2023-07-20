import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from sched_env.agent.ppo import *
from sched_env.env import GymSchedulerEnv


def init_env(workload_path, args):
    customEnv = GymSchedulerEnv(
        flatten_observation=True,
        workload_file=workload_path,
        skip=args.skip,
        job_score_type=args.score_type,
        batch_job_slice=args.batch_job_slice,
    )
    check_env(customEnv)
    return customEnv


def init_dir_from_args(args):
    score_type_dict = {0: 'bsld', 1: 'wait_time', 2: 'turnaround_time', 3: 'resource_utilization'}
    workload_name = args.workload.split('/')[-1].split('.')[0]
    current_dir = os.getcwd()

    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, args.log_dir)
    model_dir = f'{args.model_dir}/{score_type_dict[args.score_type]}/{workload_name}/'
    # model_dir = os.path.join(current_dir, model_dir)
    return model_dir, log_data_dir, workload_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./dataset/HPC2N-2002-2.2-cln.swf',
                        help='Path to workload file to be used for training')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model_dir', type=str, default='./trained_models', help='Path to save trained model')
    parser.add_argument('--log_dir', type=str, default='./data/logs/SDSC-SP2', help='Path to save log data')
    parser.add_argument('--gamma', type=float, default=1, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda coefficient in GAE')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity (0: no output, 1: info, 2: debug)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, ...)')
    parser.add_argument('--rollout_steps', type=int, default=5000, help='Number of environment steps in each rollout')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Total number of rollouts to collect during training')
    parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of training epochs over each rollout sequence')
    parser.add_argument('--trained_model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--stats_window_size', type=int, default=5)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--attn', type=bool, default=False)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--actor_model', type=str, default='kernel')
    parser.add_argument('--critic_model', type=str, default='critic_lg')
    args = parser.parse_args()

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(args)
    # create environment
    env = init_env(workload_file, args)
    if args.trained_model is not None:
        model = PPO.load(args.trained_model, env=env)
    else:
        available_models = CustomTorchModel.list_models()
        if args.actor_model in available_models and args.critic_model in available_models:
            policy_args = {'attn': args.attn, 'actor_model': args.actor_model, 'critic_model': args.critic_model}
        else:
            print(
                f"Invalid model name: {args.actor_model} or {args.critic_model} not in available models: {', '.join(available_models)}")
            print(f"Using default model: kernel and critic_lg")
            policy_args = {}

        model = PPO(CustomActorCriticPolicy, env, learning_rate=args.lr,
                    seed=args.seed, n_epochs=args.epochs,
                    n_steps=args.rollout_steps, gamma=args.gamma,
                    clip_range=args.clip_range, gae_lambda=args.gae_lambda,
                    target_kl=args.target_kl, policy_kwargs=policy_args, tensorboard_log=log_data_dir,
                    normalize_advantage=False, device=args.device, verbose=args.verbose,
                    stats_window_size=args.stats_window_size)
        env_steps = args.rollout_steps * args.num_rollouts
        print(":AGENT-PPO: Learning")
        model.learn(400000, progress_bar=True)
        model.save(f"{model_dir}ppo_HPC")
        print(f"Trained model saved at: {model_dir}ppo_HPC")
