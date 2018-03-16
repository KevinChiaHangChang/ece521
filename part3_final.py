import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255.
    data = np.reshape(data, [-1, 32*32])

    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], data[rnd_idx[trBatch+1:trBatch + validBatch],:], data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], target[rnd_idx[trBatch+1:trBatch + validBatch], task], target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

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


def reshape_target(target,num):
    x = np.zeros((len(target), num))
    x[np.arange(len(target)), target] = 1
    return x

def optimal_l_rate():

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

    trainTarget = reshape_target(trainTarget,10)
    validTarget = reshape_target(validTarget,10)
    testTarget = reshape_target(testTarget,10)

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
            sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
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

    trainTarget = reshape_target(trainTarget,10)
    validTarget = reshape_target(validTarget,10)
    testTarget = reshape_target(testTarget,10)

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[valid_size,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[test_size,-1])

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

    # specify learning rate
    each_l_rate = 0.001
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
        sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
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
    f.tight_layout()
    axarr[0].plot(train_error_list, label="training set cross-entropy loss")
    axarr[0].plot(valid_error_list, label="validation set cross-entropy loss")
    axarr[0].set_title("Cross-Entropy Loss")
    axarr[0].legend()
    axarr[1].plot(train_accuracy_list, label="training set classification accuracy")
    axarr[1].plot(valid_accuracy_list, label="validation set classification accuracy")
    axarr[1].set_title("Classification Accuracy")
    axarr[1].legend()
    plt.savefig("part2_2_1.png")

    test_error = sess.run(loss, feed_dict={X: testData, y_target: testTarget, W_lambda: each_W_lambda})
    test_accuracy = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget})
    print("Test error: %f " % test_error)
    print("Test accuracy: %f " % test_accuracy)

def optimal_l_rate_facescrub():

    # load data
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy','target.npy', 0) # operation = 0 for face recognition

    # divide data into batches
    batch_size = 300

    if batch_size == 300:
        trainData = np.concatenate((trainData[:],trainData[:153]))
        trainTarget = np.concatenate((trainTarget[:],trainTarget[:153]))

    epoch_size = len(trainData)
    valid_size = len(validData)
    test_size = len(testData)
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)

    # flatten training data
    trainData = np.reshape(trainData,[epoch_size,-1])
    # flatten validation data
    validData = np.reshape(validData,[valid_size,-1])
    # flatten test data
    testData = np.reshape(testData,[test_size,-1])

    trainTarget = reshape_target(trainTarget,6)
    validTarget = reshape_target(validTarget,6)
    testTarget = reshape_target(testTarget,6)

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[valid_size,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[test_size,-1])

    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 6], stddev=0.35, seed=521), name="weights")
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
    # W_lambdas = [0.0, 0.001, 0.1, 1]
    each_W_lambda = 0.0

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
            sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
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

def optimal_W_lambda_facescrub():

    # load data
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy','target.npy', 0) # operation = 0 for face recognition

    # divide data into batches
    batch_size = 300

    if batch_size == 300:
        trainData = np.concatenate((trainData[:],trainData[:153]))
        trainTarget = np.concatenate((trainTarget[:],trainTarget[:153]))

    epoch_size = len(trainData)
    valid_size = len(validData)
    test_size = len(testData)
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)

    # flatten training data
    trainData = np.reshape(trainData,[epoch_size,-1])
    # flatten validation data
    validData = np.reshape(validData,[valid_size,-1])
    # flatten test data
    testData = np.reshape(testData,[test_size,-1])

    trainTarget = reshape_target(trainTarget,6)
    validTarget = reshape_target(validTarget,6)
    testTarget = reshape_target(testTarget,6)

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[valid_size,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[test_size,-1])

    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 6], stddev=0.35, seed=521), name="weights")
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
    # l_rates = [0.005, 0.001, 0.0001]
    each_l_rate = 0.005
    # specify weight decay coefficient
    W_lambdas = [0.0, 0.001, 0.1, 1]

    valid_error_list = []
    valid_accuracy_list = []

    for each_W_lambda in W_lambdas:

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
            sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
            if batch_idx == numBatches-1:
                # train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
                # train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
                valid_error = sess.run(loss, feed_dict={X: validData, y_target: validTarget, W_lambda: each_W_lambda})
                valid_accuracy = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
                # train_error_list.append(train_error)
                # train_accuracy_list.append(train_accuracy)
                # valid_error_list.append(valid_error)
                # valid_accuracy_list.append(valid_accuracy)
            if step%1000==0:
                print("Step: %d " % step)

        validation_error = sess.run(loss, feed_dict={X: validData, y_target: validTarget, W_lambda: each_W_lambda})
        validation_accuracy = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
        valid_error_list.append(validation_error)
        valid_accuracy_list.append(validation_accuracy)

    best_W_lambda = W_lambdas[valid_accuracy_list.index(max(valid_accuracy_list))]
    print(valid_error_list)
    print(valid_accuracy_list)
    print("Best weight decay coefficient: %f" % best_W_lambda)


def q3part2():

    # best learning rate = 0.005
    # best weight decay coefficient = 0.0

    # load data
    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy','target.npy', 0) # operation = 0 for face recognition

    # divide data into batches
    batch_size = 300

    if batch_size == 300:
        trainData = np.concatenate((trainData[:],trainData[:153]))
        trainTarget = np.concatenate((trainTarget[:],trainTarget[:153]))

    epoch_size = len(trainData)
    valid_size = len(validData)
    test_size = len(testData)
    numBatches = epoch_size/batch_size
    print("Number of batches: %d" % numBatches)

    # flatten training data
    trainData = np.reshape(trainData,[epoch_size,-1])
    # flatten validation data
    validData = np.reshape(validData,[valid_size,-1])
    # flatten test data
    testData = np.reshape(testData,[test_size,-1])

    trainTarget = reshape_target(trainTarget,6)
    validTarget = reshape_target(validTarget,6)
    testTarget = reshape_target(testTarget,6)

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[valid_size,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[test_size,-1])

    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 6], stddev=0.35, seed=521), name="weights")
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
    # l_rates = [0.005, 0.001, 0.0001]
    each_l_rate = 0.005
    # specify weight decay coefficient
    # W_lambdas = [0.0, 0.001, 0.1, 1]
    each_W_lambda = 0.0

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
        sess.run([train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
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
    f.tight_layout()
    axarr[0].plot(train_error_list, label="training set cross-entropy loss")
    axarr[0].plot(valid_error_list, label="validation set cross-entropy loss")
    axarr[0].set_title("Cross-Entropy Loss")
    axarr[0].legend()
    axarr[1].plot(train_accuracy_list, label="training set classification accuracy")
    axarr[1].plot(valid_accuracy_list, label="validation set classification accuracy")
    axarr[1].set_title("Classification Accuracy")
    axarr[1].legend()
    plt.savefig("part2_2_2.png")

    test_error = sess.run(loss, feed_dict={X: testData, y_target: testTarget, W_lambda: each_W_lambda})
    test_accuracy = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget})
    print("Test error: %f " % test_error)
    print("Test accuracy: %f " % test_accuracy)



if __name__ == '__main__':
    # optimal_l_rate()
    # optimal_l_rate_facescrub()
    # optimal_W_lambda_facescrub()
    # q3part1()
    q3part2()

