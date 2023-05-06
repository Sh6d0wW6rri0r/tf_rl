import tensorflow as tf

class Critic(tf.keras.Model):
	def __init__(self,env):
		super().__init__()
		self.model=tf.keras.Sequential()
		self.model.add(tf.keras.layers.Input(shape=env.observation_space.shape))
		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(64,activation='relu'))
		self.model.add(tf.keras.layers.Dense(64,activation='relu'))
		self.model.add(tf.keras.layers.Dense(1, activation = None))

	def call(self, input_data):
		v=self.model(input_data)
		return v