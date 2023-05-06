import numpy as np 
import gym
import tensorflow as tf

from DDQN import DDQN
from DQN import DQN
from ActorCritic import ActorCritic

class Agent():
	def __init__(self,env,algo='DDQN',actions_size=-1,max_steps=100):
		self.env=env
		self.algorithm = []
		self.alg_name=algo
		self.actions_size=0
		self.max_steps=max_steps
		if actions_size==-1:
			self.actions_size=self.env.action_space.n
		else:
			self.actions_size=actions_size
		self.delayed_learning=False
		self.use_epsilon = False
		if algo=='DDQN':
			self.algorithm=DDQN(self.env,actions_size)
			self.delayed_learning=True
			self.use_epsilon=True
		if algo=='DQN':
			self.algorithm=DQN(self.env,actions_size)
			self.delayed_learning=True
			self.use_epsilon=True
		if algo=='ActorCritic':
			self.algorithm=ActorCritic(self.env,actions_size)
			self.delayed_learning=False
			self.use_epsilon=False
		self.epsilon = 1.0
		self.min_epsilon = 0.2
		self.epsilon_decay = 0.00005

	def update_epsilon(self):
		if self.use_epsilon==True:
			self.epsilon=self.epsilon-self.epsilon_decay
			if self.epsilon<self.min_epsilon:
				self.epsilon=self.min_epsilon
				

	  
	def act(self,state):
		act = 0
		if self.use_epsilon==True:
			if np.random.rand() <= self.epsilon:
				act = np.random.randint(self.actions_size)
			else:
				q_values=self.algorithm.predict(state)
				act = tf.argmax(q_values, axis=1)[0].numpy()
		else:
			q_values=self.algorithm.predict(state)
			act = tf.argmax(q_values, axis=1)[0].numpy()
		return act
				
	def check(self):
		total_reward = 0
		state , _ = self.env.reset()
		done = False
		st = 0
		while not done:
			q_values=self.algorithm.predict(state)
			action = tf.argmax(q_values, axis=1)[0].numpy()
			next_state, reward, done, _ , _ = self.env.step(action)
			state = next_state
			total_reward += reward
			st+=1
			if st>self.max_steps:
				done=True
		return total_reward
		
		
	def solve(self,reward_solved):	
		done = False
		state , _ = self.env.reset()
		rw = 0
		ep = 0
		st=0
		max_rew =0
		while max_rew<=reward_solved:
			action = self.act(state)
			next_state, reward, done, _ , _ = self.env.step(action)
			if self.delayed_learning==True:
				self.algorithm.add_memory(state,action,reward,next_state,done)
			else:
				self.algorithm.learn(state,action,reward,next_state,done)
			state = next_state
			rw+=reward
			st+=1
			if st>=self.max_steps:
				done=True
			if done:
				if self.delayed_learning==True:
					self.algorithm.learn()
				ep+=1
				print("Episode ",ep," completed in ",st," steps, with reward : ",rw)
				if self.use_epsilon==True:
					print("Epsilon : ",self.epsilon)
				st=0
				rw=0
				self.update_epsilon()
				max_rew = self.check()
				print("Test reward: ",max_rew)
				state , _ =self.env.reset()
		print("After training the agent reward on test is ",self.check())
