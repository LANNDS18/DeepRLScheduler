#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from gym.envs.registration import register

from .deeprm_env import DeepRmEnv


logger = logging.getLogger(__name__)

register(
    id='DeepRM-v0',
    nondeterministic=False,
    entry_point=f'sched_env.envs.deeprm_env:{DeepRmEnv.__name__}',
)
