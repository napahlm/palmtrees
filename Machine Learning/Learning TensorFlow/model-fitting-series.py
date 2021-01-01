import tensorflow as tf
import numpy as np

keras = tf.keras

# model with a layer with a single neuron being fed a single value, x
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# compiling the model with an optimizer and a loss function
# the optimizer makes a guess, and the loss function evaluates the guess
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=5000)

# guessing the y-value for a given x
print(model.predict([10.0]))