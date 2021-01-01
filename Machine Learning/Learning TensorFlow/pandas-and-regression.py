from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# checks for unknown values
dataset.isna().sum()
# dropping the unknown values
dataset = dataset.dropna()

# Origin is a categorical column
# Want to convert it to one hot column
origin = dataset.pop('Origin')

# Sets a value of one from each country spot and zero in the others
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()

# Splitting the data into several pieces: train and test
# Generalization: A model doing well on data never seen before
train_dataset = dataset.sample(frac=0.8, random_state=0)  # 80% of the data for training
test_dataset = dataset.drop(train_dataset.index)

"""
Seaborn and its pairplot features

Pairplot:
    Gives plots of joint distributions of given columns.
    
Seeing patterns in these plots gives an indicator of dependancy in these different features.
"""

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# Summarize some stats

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# train_stats

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

# Normalization of our features (0 to 1)
def norm(x):
    return (x - train_stats['mean']) / (train_stats['std'])


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Building the model with keras and 3 dense layers

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)  # no activation, linear activation
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # Mean Squared Error as the loss function
    model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])
    return model

model = build_model()

# Inspecting it!

model.summary()

# Try it! (Without training!)

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
# example_result

""" Training my model with 1000 epochs """

# Displaying training progress by printing a dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000

# Splits of 20% of the data for validation
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

# Looking at the history

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])

plot_history(history)

# The loss function is increasing! Classic overfitting

# Early stopping: Stops training the model when the model stops improving
# Decide:
# What to look for (val_loss) and how much patience?

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

# Time to check out performance

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Predictions

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# Error distribution

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

"""
Conclusion

- MSE common for regression
- MAE is a common regression metric
- Avoid overfitting with few hidden layers
- Or just early stop

"""