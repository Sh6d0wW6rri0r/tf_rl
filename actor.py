import tensorflow as tf

class Actor(tf.keras.Model):
	def __init__(self,env,actions_size=-1):
		super().__init__()
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Input(shape=env.observation_space.shape))
		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(64,activation='relu'))
		self.model.add(tf.keras.layers.Dense(64,activation='relu'))
		if actions_size==-1:
			self.model.add(tf.keras.layers.Dense(env.action_space.n,activation='softmax'))
		else:
			self.model.add(tf.keras.layers.Dense(actions_size,activation='softmax'))
				
	def call(self, input_data):
		a=self.model(input_data)
		return a