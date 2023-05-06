import gym
import numpy as np
import tensorflow as tf
from gym_minigrid.wrappers import ImgObsWrapper,ReseedWrapper
from Agent import Agent

env = gym.make('MiniGrid-FourRooms-v0')
env = ReseedWrapper(env)
env = ImgObsWrapper(env)



ag = Agent(env,'ActorCritic',3,100)
ag.solve(0.80)
print("Reward : ",ag.check())
