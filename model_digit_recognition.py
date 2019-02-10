from skimage import transform
from skimage import data
import skimage as sk
from skimage import util
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random
import tensorflow as tf
import cv2
from skimage.color import rgb2gray
from skimage import exposure


image_size=28*2
X_train = []
y_train = []

X_test = []
y_test = []

for p in os.listdir(".../pics"):
    rand1=random.random()
    #разбиваем выборку на тестовую и контрольную в соотношении 90%/10%
    if rand1<0.9:
        X_train.append(data.imread(os.path.join(".../pics/",p)))
        y_train.append(int(p[2]))
        # print(new_img_array)
    else:
        X_test.append(data.imread(os.path.join(".../pics/",p)))
        y_test.append(int(p[2]))
    #Data Augmentation
    for i in range(1,5):
        aug_img_array=sk.util.random_noise(data.imread(os.path.join(".../pics/",p)))
        rand2 = random.random()
        if rand2<0.9:
            print(i)
            X_train.append(aug_img_array)
            y_train.append(int(p[2]))
        else:
            X_test.append(sk.util.random_noise(aug_img_array))
            y_test.append(int(p[2]))

images=X_train
labels=y_train

images_array = np.array(X_train)
labels_array = np.array(y_train)

# Change image size
images32 = [transform.resize(image, (image_size, image_size)) for image in images]
images32 = np.array(images32)

images32 = rgb2gray(np.array(images32))
print(images32.shape)

x = tf.placeholder(dtype = tf.float32, shape = [None, image_size, image_size])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(101):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})
        print('DONE WITH EPOCH')

predicted = sess.run([correct_pred], feed_dict={x: images32})[0]

# Load the test data
test_images = np.array(X_test)
test_labels = np.array(y_test)

# Transform the images to 28 by 28 pixels
test_images_ = [transform.resize(image, (image_size, image_size)) for image in test_images]

# Convert to grayscale
test_images_ = rgb2gray(np.array(test_images_))

# Run predictions against the full test set.
predicted_test = sess.run([correct_pred], feed_dict={x: test_images_})[0]

# Calculate correct matches
match_count_test = sum([int(y == y_) for y, y_ in zip(test_labels, predicted_test)])
match_count = sum([int(y == y_) for y, y_ in zip(labels, predicted)])

# Calculate the accuracy
accuracy_test = match_count_test / len(test_labels)
accuracy = match_count / len(labels)

print(test_labels)
print(predicted_test)
# Print the accuracy
print("Accuracy train: {:.3f}".format(accuracy))
print("Accuracy test: {:.3f}".format(accuracy_test))
