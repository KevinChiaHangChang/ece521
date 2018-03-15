import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def data_gen():

    with np.load("notMNIST.npz") as data:
        data = np.load("notMNIST.npz")
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def q2part1():

    # load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = data_gen()

    # divide data into batches
    epoch_size = len(trainData)
    batch_size = 500
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)


    # flatten training data
    trainData = np.reshape(trainData,[epoch_size,-1])
    # flatten validation data
    validData = np.reshape(validData,[100,-1])
    # flatten test data
    testData = np.reshape(testData,[145,-1])

    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 1], stddev=0.35, seed=521), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, name="input_x")
    y_target = tf.placeholder(tf.float32, name="target_y")

    # graph definition
    y_pred = tf.matmul(X,W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_target)/2.0, name="logits_loss")
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2.0
    loss = logits_loss + W_decay

    # accuracy definition
    y_pred_sigmoid = tf.sigmoid(y_pred)
    accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.cast(tf.greater(y_pred_sigmoid,0.5),tf.float32), y_target ), tf.float32) )

    # training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    # specify learning rate
    each_l_rate = 0.005
    # specify weight decay coefficient
    each_W_lambda = 0.01

    # initialize session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(W)
    sess.run(b)

    train_error_list = []
    train_accuracy_list = []
    valid_error_list = []
    valid_accuracy_list = []

    for step in range(0,5000):
        batch_idx = step%numBatches
        trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        _, train_W, train_b, train_yhat = sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
        if batch_idx == numBatches-1:
            train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
            train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
            valid_error = sess.run(loss, feed_dict={X: validData, y_target: validTarget, W_lambda: each_W_lambda})
            valid_accuracy = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
            train_error_list.append(train_error)
            train_accuracy_list.append(train_accuracy)
            valid_error_list.append(valid_error)
            valid_accuracy_list.append(valid_accuracy)
        if step%1000==0 and step > 0:
            print("Step: %d " % step)
            print("Training error: %f " % train_error_list[-1])
            print("Validation error: %f " % valid_error_list[-1])


    # plot image
    plt.clf()
    f, axarr = plt.subplots(2)
    axarr[0].plot(train_error_list, label="training set cross-entropy loss")
    axarr[0].plot(valid_error_list, label="validation set cross-entropy loss")
    axarr[0].set_title("cross-entropy loss")
    axarr[0].legend()
    axarr[1].plot(train_accuracy_list, label="training set classification accuracy")
    axarr[1].plot(valid_accuracy_list, label="validation set classification accuracy")
    axarr[1].set_title("classification accuracy")
    axarr[1].legend()
    plt.savefig("part2_1_1.png")

if __name__ == '__main__':
    q2part1()

