import tensorflow as tf
import tensorflow_addons as tfa
import keras
import cv2
import numpy as np
from facenet import face_net_network
from crop_image import processing_data
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("path1")
args = parser.parse_args()

X,Y= processing_data(args.path1)
x_train, x_test, y_train, y_test ,id_train, id_test = train_test_split(X, Y, np.array(X), test_size = 0.2)

gen_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_train, axis=-1), y_train)).repeat().shuffle(1024).batch(32)
model = face_net_network(args.model)

model.compile(optimizer= tf.keras.optimizers.Adam(lr=1e-4), loss= tfa.losses.TripletSemiHardLoss())

model.fit(gen_train, steps_per_epoch=50, epochs= 10, verbose=1)

model.save('./model/facenetrt.h5')
model.load_weights('./model/facenetrt.h5')
x_train_predict = model.predict(np.expand_dims(x_train), axis=-1)
x_test_predict = model.predict(np.expand_dims(x_test, axis=-1))
def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label
y_preds = []
for vec in x_test_predict :
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(x_train_predict , vec, y_train)
  y_preds.append(y_pred)

print('Accuracy in test: {}'.format(accuracy_score(y_preds, y_test)))

from sklearn.metrics import accuracy_score
y_preds1 = []
for vec in x_train_predict:
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(x_train_predict, vec, y_train)
  y_preds1.append(y_pred)

print('Accuracy in test: {}'.format(accuracy_score(y_preds1, y_train)))
