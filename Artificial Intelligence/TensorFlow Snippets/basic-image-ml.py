import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
# a set to train the algorithm and one to test it
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# making a model
# relu: returning positive values, else is filtered
# softmax: picks biggest number in a set
model = keras.Sequential([
    # input is the image size
    keras.layers.Flatten(input_shape=(28, 28)),
    # 128 functions
    keras.layers.Dense(128, activation=tf.nn.relu),
    # the output shape is the number of categories
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# nn initialized with random numbers
# optimizer generates new parameters
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

# fitting training data
model.fit(train_images, train_labels, epochs=100)

# evaluating the results
test_loss, test_acc = model.evaluate(test_images, test_labels)

# predictions = model.predict(my_images)