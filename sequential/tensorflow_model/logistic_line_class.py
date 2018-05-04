'''
modified from project: https://github.com/aymericdamien/TensorFlow-Examples/
author: Eli Yu, Ethan Gruman
'''

from __future__ import print_function

import tensorflow as tf
import os
import PIL
from PIL import Image
import numpy as np

np.set_printoptions(threshold=np.inf,linewidth=np.inf)

def saveWeight(filename,w,weightname):
    with open(filename,"ab") as f:
        s = w.shape
        outstr = ""
        f.write("Weight\n")
        f.write(weightname+'\n')
        for dim in s:
            outstr=outstr+str(dim)+" "
        f.write(outstr[:-1]+'\n')
        w=np.reshape(w,[-1])
        w = np.array2string(w,precision=3,separator=" ",suppress_small=True)
        f.write(w[2:-1]+'\n')

def saveFC(filename,op,opname,depends):
    with open(filename,"ab") as f:
        depStr = " ".join(depends)
        f.write("Operation\n")
        f.write(op+" "+opname+" "+depStr+'\n')

def saveInput(filename,inputname,dims):
    with open(filename,"ab") as f:
        f.write("Input\n")
        f.write(inputname+'\n')
        outstr = ""
        for dim in dims:
            outstr=outstr+str(dim)+" "
        f.write(outstr[:-1])
        f.write('\n')

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 8
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 768],name='x') # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 3],name ='y') # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([768, 3]))
b = tf.Variable(tf.zeros([3]))

# Construct model
#pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
pred_temp = tf.matmul(x,W)+b
pred = tf.nn.softmax(pred_temp)
#pred = tf.Print(pred,[pred])

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    mypath = os.getcwd()
    xs = []
    ys = []
    for f in os.listdir('/home/pi/tensorflow_example/artificial-processed'):
        img = Image.open(os.path.join('/home/pi/tensorflow_example/artificial-processed/',f))
        img.load()
        img_data = np.asarray(img)/256.
        img_data= np.reshape(img_data,[-1])
        xs.append(img_data)
        fname = f.split('-')
        if('right' in fname[0]):
            ys.append([0,0,1])
        elif('left' in fname[0]):
            ys.append([1,0,0])
        else:
            ys.append([0,1,0])
    xs = np.stack(xs,axis=0)
    ys = np.stack(ys,axis=0)
    
    test_xs = xs[234:,:]
    test_ys = ys[234:,:]
    xs = xs[:234,:]
    ys = ys[:234,:]
    #print(xs.shape)
    #print(xs[0])
    #print(ys)
    #print(test_ys)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(xs.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = xs[i*batch_size:(i+1)*batch_size,:]
            batch_ys = ys[i*batch_size:(i+1)*batch_size,:]
            #print(batch_xs)
            #print(batch_ys)
            #assert(False)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    result = tf.argmax(pred, 1,name='result') 
    correct_prediction = tf.equal(result, tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test_xs, y: test_ys}))
    #print("W= ",sess.run(tf.transpose(W)))
    saveInput("test_file",'x',[1,768])
    saveWeight("test_file",sess.run(tf.transpose(W)),'W') 
    saveWeight("test_file",sess.run(b),'b') 
    saveFC("test_file","FC","FC_out",['x','W','b'])
    #saver = tf.train.Saver()
    #saver.save(sess, 'my_test_model',global_step=1000)
    test_arr = np.zeros([1,768])
    for i in range(768):
        test_arr[0,i] = i%6-2
    print(pred_temp.eval({x:test_arr}))
