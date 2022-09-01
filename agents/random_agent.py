import gym
from gym.wrappers import TimeLimit
import json
import random
import numpy as np
from novelty_bonus import NoveltyBonus
import marneo_env
import sys

rom_path = sys.argv[1]
env = gym.make('marneo/MarneoEnv-v0',
    identifier='env_random',
    rom_path=rom_path)
env = NoveltyBonus(env)
env = TimeLimit(env, max_episode_steps=300)
try:
    obs = env.reset()
    while True:
        action = random.randint(0, env.unwrapped.action_space.n-1)
        obs, reward, done, info = env.step(action)
        print(reward)
        if done:
            obs = env.reset()
finally:
    env.close()