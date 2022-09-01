import gym
from gym.wrappers import TimeLimit
import json
import random
import numpy as np
import marneo_env
import sys
import cv2

from collections import deque
import random

from comparison_buffer import ComparisonBuffer

rom_path = sys.argv[1]
env = gym.make('marneo/MarneoEnv-v0',
    identifier='env_random',
    rom_path=rom_path,
    port=14000)
# env = TimeLimit(env, max_episode_steps=600)
cmp_buff = ComparisonBuffer(1000)
try:
    obs = env.reset()
    while True:
        action = random.randint(0, env.unwrapped.action_space.n-1)
        obs, reward, done, info = env.step(action)
        pixel_representation = info.get('pixel_representation')
        try:
            if pixel_representation:
                image = cv2.imread(pixel_representation, 0)
                cmp_buff.push(image)
                if len(cmp_buff) > 20:
                    similarity = cmp_buff.calculate_average_similarity(10, image)
                    reward = 1 - similarity
        except:
            reward = 0.0
        _ = env.step(action)
        if done:
            obs = env.reset()
finally:
    env.close()