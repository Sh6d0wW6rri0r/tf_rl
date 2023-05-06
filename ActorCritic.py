import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import gym

from Actor import Actor
from Critic import Critic
from Algorithm import Algorithm

class ActorCritic(Algorithm):
	def __init__(self,env,actions_size=-1):
		self.a_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.c_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.env=env
		self.actor = Actor(self.env)
		self.critic = Critic(self.env)
		self.actions_size=0
		if actions_size==-1:
			self.actions_size=env.action_space.n
		else:
			self.actions_size=actions_size

		
	def predict(self,state):
		q_values=self.actor(tf.expand_dims(state, 0))
		return q_values

					
	def actor_loss(self, prob, action, td):
		dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
		log_prob = dist.log_prob(action)
		loss = -log_prob*td
		return loss
					
	def learn(self, state,action,reward,next_state,done):
		state = np.array([state])
		next_state = np.array([next_state])
		with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
			p = self.actor(state, training=True)
			v =  self.critic(state,training=True)
			vn = self.critic(next_state, training=True)
			td = reward + 0.99*vn*(1-int(done)) - v
			a_loss = self.actor_loss(p, action, td)
			c_loss = td**2
		grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
		grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
		self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
		self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
		return a_loss, c_loss
		
