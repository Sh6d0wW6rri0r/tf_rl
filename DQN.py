
import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import gym

from Actor import Actor
from Algorithm import Algorithm

class DQN(Algorithm):
	def __init__(self,env,actions_size=-1):
		self.env=env
		self.q_network = Actor(self.env,actions_size)
		self.actions_size=0
		if actions_size==-1:
			self.actions_size=self.env.action_space.n
		else:
			self.actions_size=actions_size
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.q_network.compile(loss='mse',optimizer=self.optimizer)
		self.nb_train=0
		self.batch_size=128
		self.memory_size=50000
		self.states = np.zeros((self.memory_size, *(env.observation_space.shape)), dtype=np.float32)
		self.actions = np.zeros((self.memory_size), dtype=np.int32)
		self.rewards = np.zeros((self.memory_size), dtype=np.float32)
		self.next_states = np.zeros((self.memory_size, *(env.observation_space.shape)), dtype=np.float32)
		self.dones = np.zeros((self.memory_size), dtype=np.bool)
		self.last_memory = -1
		self.insert_memory = -1
		self.gamma = 0.99
		
	def predict(self,state):
		q_values=self.q_network(tf.expand_dims(state, 0))
		return q_values

	def add_memory(self,state,action,reward,next_state,done):
		self.insert_memory+=1
		self.last_memory+=1
		if self.last_memory>=self.memory_size:
			self.last_memory=self.memory_size
		if self.insert_memory>=self.memory_size:
			self.insert_memory=0
		self.states[self.insert_memory]=state
		self.actions[self.insert_memory]=action
		self.rewards[self.insert_memory]=reward
		self.next_states[self.insert_memory]=next_state
		self.dones[self.insert_memory]=1-int(done)
	
	def get_memory(self):
		batch_indices = np.random.choice(self.last_memory,self.batch_size,replace=False)
		st = self.states[batch_indices]
		ac = self.actions[batch_indices]
		rw = self.rewards[batch_indices]
		ns = self.next_states[batch_indices]
		dn = self.dones[batch_indices]
		return st,ac,rw,ns,dn
		
			
	def learn(self):
		if self.last_memory < self.batch_size:
			return
		st, act, rew, n_st, don = self.get_memory()
		target_q_values = self.q_network(n_st)
		max_target_q_values = tf.reduce_max(target_q_values, axis=1)
		target_q_values = rew + (1 - don) * self.gamma * max_target_q_values
		with tf.GradientTape() as tape:
			q_values = self.q_network(st)
			one_hot_actions = tf.one_hot(act, self.actions_size)
			q_values_for_actions = tf.reduce_sum(one_hot_actions * q_values, axis=1)
        	# Calculate the loss and gradients
			loss = tf.keras.losses.MSE(target_q_values, q_values_for_actions)
			gradients = tape.gradient(loss, self.q_network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

