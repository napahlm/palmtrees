import tensorflow as tf
import imageio

# Importing an example image to work with
img_raw = tf.io.read_file('example-image.png')
img = tf.image.decode_image(img_raw)
print('Image shape: ', img.shape)

img = imageio.imread('example-image.png')
print('Image shape: ', img.shape)
