import logging

import gym
from gym.envs.registration import register

from deepRL_scheduler.sched_env.gym_env.gym_env import DeepRmEnv

logger = logging.getLogger(__name__)

register(
    id='sched-v0',
    nondeterministic=False,
    entry_point=f'sched_env.gym_env.gym_env:{DeepRmEnv.__name__}',
)


env = gym.make('sched-v0', use_raw_state=True)
env.reset()

for _ in range(200):
  env.render()
  observation, reward, done, info = env.step(env.action_space.sample())
env.close()