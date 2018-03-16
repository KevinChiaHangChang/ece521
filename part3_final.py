import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def data_gen():

    data = np.load("notMNIST.npz")
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    return trainData, trainTarget, validData, validTarget, testData, testTarget


def reshape_target(target):
    x = np.zeros((len(target), 10))
    x[np.arange(len(target)), target] = 1
    return x

def q3part1():

    # load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = data_gen()

    # divide data into batches
    epoch_size = len(trainData)
    valid_size = len(validData)
    test_size = len(testData)
    batch_size = 500
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)

    # flatten training data
    trainData = np.reshape(trainData,[epoch_size,-1])
    # flatten validation data
    validData = np.reshape(validData,[valid_size,-1])
    # flatten test data
    testData = np.reshape(testData,[test_size,-1])

    trainTarget = reshape_target(trainTarget)
    validTarget = reshape_target(validTarget)
    testTarget = reshape_target(testTarget)

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[valid_size,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[test_size,-1])

    print("Train data size: %d x %d" % (trainData.shape[0],trainData.shape[1]))
    print("Validation data size: %d x %d" % (validData.shape[0],validData.shape[1]))
    print("Test data size: %d x %d" % (testData.shape[0],testData.shape[1]))

    print("Train target size: %d" % (trainTarget.shape[0]))
    print("Validation target size: %d" % (validTarget.shape[0]))
    print("Test target size: %d" % (testTarget.shape[0]))


    print(np.shape(trainTarget))


    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 10], stddev=0.35, seed=521), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, name="input_x")
    y_target = tf.placeholder(tf.float32, name="target_y")


    # graph definition
    y_pred = tf.matmul(X,W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    logits_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_target)/2.0, name="logits_loss")
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2.0
    loss = logits_loss + W_decay

    # accuracy definition
    y_pred_sigmoid = tf.sigmoid(y_pred)
    correct_predictions = tf.equal(tf.argmax(y_pred_sigmoid, 1), tf.argmax(y_target, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), 0)


    # training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    # specify learning rates
    l_rates = [0.005, 0.001, 0.0001]
    # specify weight decay coefficient
    each_W_lambda = 0.01

    # train_error_list = []
    # train_accuracy_list = []
    # valid_error_list = []
    # valid_accuracy_list = []
    train_error_list = []
    train_accuracy_list = []

    for each_l_rate in l_rates:

        # initialize session
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        sess.run(W)
        sess.run(b)

        for step in range(0,5000):
            batch_idx = step%numBatches
            trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            # print("Train data size: %d x %d" % (trainDataBatch.shape[0],trainDataBatch.shape[1]))
            # print("Train target size: %d" % (trainTargetBatch.shape[0]))
            _, currentW, currentb, yhat = sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
            if batch_idx == numBatches-1:
                train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
                train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
                # valid_error = sess.run(loss, feed_dict={X: validData, y_target: validTarget, W_lambda: each_W_lambda})
                # valid_accuracy = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
                # train_error_list.append(train_error)
                # train_accuracy_list.append(train_accuracy)
                # valid_error_list.append(valid_error)
                # valid_accuracy_list.append(valid_accuracy)
            if step%1000==0:
                print("Step: %d " % step)





        # print (yhat)
        training_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
        training_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
        train_error_list.append(training_error)
        train_accuracy_list.append(training_accuracy)

    best_l_rate = l_rates[train_error_list.index(min(train_error_list))]
    print(train_error_list)
    print(train_accuracy_list)
    print("Best learning rate: %f " % best_l_rate)

    # # plot image
    # plt.clf()
    # f, axarr = plt.subplots(2)
    # axarr[0].plot(train_error_list, label="training set cross-entropy loss")
    # axarr[0].plot(valid_error_list, label="validation set cross-entropy loss")
    # axarr[0].set_title("Cross-Entropy Loss")
    # axarr[0].legend()
    # axarr[1].plot(train_accuracy_list, label="training set classification accuracy")
    # axarr[1].plot(valid_accuracy_list, label="validation set classification accuracy")
    # axarr[1].set_title("Classification Accuracy")
    # axarr[1].legend()
    # plt.savefig("part2_1_1.png")

if __name__ == '__main__':
    q3part1()

