import tensorflow as tf 
import tensorflow_probability as tfp
import tensorflow.keras.metrics as kls
import numpy as np 
import gym

class Algorithm():
	def __init__(self,env):
		print("Not to be instanciated !")
		
	def predict(self,state):
		print("Not to be instanciated !")
			
	def learn(self, states, actions,  adv , old_probs, discnt_rewards):
		print("Not to be instanciated !")
		