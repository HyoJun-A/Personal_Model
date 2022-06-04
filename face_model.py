import tensorflow as tf
import os
import numpy as np
import PIL

np.set_printoptions(linewidth=1000)

def save_model(h5_path, model_path):
    model = tf.keras.Sequential()    
    model.add(tf.keras.layers.Dense(10, activation='softmax', input_shape=[784]))    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),                  
                  loss=tf.keras.losses.sparse_categorical_crossentropy,                 
                  metrics=['acc'])    
    mnist = input_data.read_data_sets('mnist')    
    model.fit(mnist.train.images, mnist.train.labels,              
              validation_data=[mnist.validation.images, mnist.validation.labels],              
              epochs=15, batch_size=128, verbose=0)
    with open(model_path, 'wb') as f:        
        f.write(flat_data)

save_model('./model/mnist.h5', './model/mnist.tflite')
