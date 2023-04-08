import gym
from gym import spaces

import numpy as np

from tetris_gym.utils.board_utils import get_heights, get_bumps_from_heights


class ExtendedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = spaces.Dict({
            "board": env.observation_space["board"],
            "piece": env.observation_space["piece"],
            "heights": spaces.Box(
                low=0,
                high=env.height,
                shape=(env.width,),
                dtype=int,
            ),
            "bumps": spaces.Box(
                low=0,
                high=env.height,
                shape=(env.width - 1,),
                dtype=int,
            )
        })
       
    def observation(self, obs):

        board = obs["board"]
        piece = obs["piece"]

        heights = get_heights(board)

        bumps = get_bumps_from_heights(heights)

        obs = {
            "board": board,
            "piece": piece,
            "heights": heights,
            "bumps": bumps
        }
        
        return obs


class HeightsObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "heights": env.observation_space["heights"],
            "bumps": env.observation_space["bumps"],
            "piece": env.observation_space["piece"]
        })

    def observation(self, obs):

        return {
            "heights": obs["heights"],
            "bumps": obs["bumps"],
            "piece": obs["piece"]
        }