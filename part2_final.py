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

def optimal_l_rate():

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

    # reshape training target
    trainTarget = np.reshape(trainTarget,[epoch_size,-1])
    # reshape validation target
    validTarget = np.reshape(validTarget,[100,-1])
    # reshape test target
    testTarget = np.reshape(testTarget,[145,-1])

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

    # specify learning rates
    l_rates = [0.005, 0.001, 0.0001]
    # specify weight decay coefficient
    each_W_lambda = 0.0

    train_error_list = []

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
        
        training_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
        train_error_list.append(training_error)

    best_l_rate = l_rates[train_error_list.index(min(train_error_list))]
    print(train_error_list)
    print("Best learning rate: %f " % best_l_rate)

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

    last_training_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
    last_training_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
    print("Last training error: %f" % last_training_error)
    print("Last training accuracy: %f" % last_training_accuracy)

    test_error = sess.run(loss, feed_dict={X: testData, y_target: testTarget, W_lambda: each_W_lambda})
    test_accuracy = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget})
    print("Test error: %f" % test_error)
    print("Test accuracy: %f" % test_accuracy)

    # plot image
    plt.clf()
    f, axarr = plt.subplots(2)
    axarr[0].plot(train_error_list, label="training set cross-entropy loss")
    axarr[0].plot(valid_error_list, label="validation set cross-entropy loss")
    axarr[0].set_ylabel("Cross-Entropy Loss")
    axarr[0].set_xlabel("Number of Epoch")
    axarr[0].legend()
    axarr[1].plot(train_accuracy_list, label="training set classification accuracy")
    axarr[1].plot(valid_accuracy_list, label="validation set classification accuracy")
    axarr[1].set_ylabel("Classification Accuracy")
    axarr[1].set_xlabel("Number of Epoch")
    axarr[1].legend()
    f.tight_layout()
    f.suptitle("Two-Class notMNIST")
    f.subplots_adjust(top=0.88)
    plt.savefig("part2_1_1.png")

def q2part2():

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

    # training mechanism
    sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    sgd_train = sgd_optimizer.minimize(loss=loss)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    adam_train = adam_optimizer.minimize(loss=loss)

    # specify learning rate
    each_l_rate = 0.001
    # specify weight decay coefficient
    each_W_lambda = 0.01

    sgd_error_list = []
    adam_error_list = []

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
        sess.run([sgd_train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
        if batch_idx == numBatches-1:
            sgd_train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
            sgd_error_list.append(sgd_train_error)
        if step%1000==0 and step > 0:
            print("Step: %d " % step)
            print("SGD training error: %f " % sgd_error_list[-1])

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
        sess.run([adam_train, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: each_W_lambda})
        if batch_idx == numBatches-1:
            adam_train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
            adam_error_list.append(adam_train_error)
        if step%1000==0 and step > 0:
            print("Step: %d " % step)
            print("Adam training error: %f " % adam_error_list[-1])

    plt.clf()
    plt.plot(sgd_error_list, label="SGD training error")
    plt.plot(adam_error_list, label="Adam training error")
    plt.title("SGD vs Adam Optimizer Training Error")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig("part_2_1_2.png")

def q2part3():

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    trainData, trainTarget, validData, validTarget, testData, testTarget = data_gen()

    biasTrainData = tf.convert_to_tensor(trainData)
    biasTrainData = tf.reshape(biasTrainData, [3500, -1])
    biasVec = tf.convert_to_tensor([1 for i in range(3500)])
    biasVec = tf.cast(biasVec, tf.float64)
    biasVec = tf.reshape(biasVec, [-1, 3500])
    biasTrainData = tf.transpose(biasTrainData)
    biasTrainData = tf.concat([biasVec, biasTrainData], 0)
    biasTrainData = tf.transpose(biasTrainData)
    biasTrainTarget = tf.convert_to_tensor(trainTarget)
    biasTrainTarget = tf.cast(biasTrainTarget, tf.float64)

    biasValidData = tf.convert_to_tensor(validData)
    biasValidData = tf.reshape(biasValidData, [100, -1])
    biasVec = tf.convert_to_tensor([1 for i in range(100)])
    biasVec = tf.cast(biasVec, tf.float64)
    biasVec = tf.reshape(biasVec, [-1, 100])
    biasValidData = tf.transpose(biasValidData)
    biasValidData = tf.concat([biasVec, biasValidData], 0)
    biasValidData = tf.transpose(biasValidData)
    biasValidTarget = tf.convert_to_tensor(validTarget)
    biasValidTarget = tf.cast(biasValidTarget, tf.float64)

    biasTestData = tf.convert_to_tensor(testData)
    biasTestData = tf.reshape(biasTestData, [145, -1])
    biasVec = tf.convert_to_tensor([1 for i in range(145)])
    biasVec = tf.cast(biasVec, tf.float64)
    biasVec = tf.reshape(biasVec, [-1, 145])
    biasTestData = tf.transpose(biasTestData)
    biasTestData = tf.concat([biasVec, biasTestData], 0)
    biasTestData = tf.transpose(biasTestData)
    biasTestTarget = tf.convert_to_tensor(testTarget)
    biasTestTarget = tf.cast(biasTestTarget, tf.float64)

    W_norm = tf.matmul(tf.transpose(biasTrainData), biasTrainData)
    W_norm = tf.matrix_inverse(W_norm)
    W_norm = tf.matmul(W_norm, tf.transpose(biasTrainData))
    W_norm = tf.matmul(W_norm, biasTrainTarget)

    epoch_size = len(trainData)
    batch_size = 500
    numBatches = epoch_size / batch_size

    # flatten training data
    trainData = np.reshape(trainData, [epoch_size, -1])
    # flatten validation data
    validData = np.reshape(validData, [100, -1])
    # flatten test data
    testData = np.reshape(testData, [145, -1])

    # reshape training target
    trainTarget = np.reshape(trainTarget, [epoch_size, -1])
    # reshape validation target
    validTarget = np.reshape(validTarget, [100, -1])
    # reshape test target
    testTarget = np.reshape(testTarget, [145, -1])

    # variable creation
    W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 1], stddev=0.35, seed=521), name="weights")
    # W = tf.Variable(tf.random_normal(shape=[trainData.shape[1], 1], stddev=0.1), name="weights")
    # W = tf.Variable(tf.zeros([trainData.shape[1], 1]), dtype=tf.float32)
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, name="input_x")
    y_target = tf.placeholder(tf.float32, name="target_y")

    # graph definition
    y_pred = tf.matmul(X, W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_target) / 2.0,
                                 name="logits_loss")
    W_decay = tf.reduce_sum(tf.multiply(W, W)) * W_lambda / 2.0
    loss = logits_loss + W_decay

    # accuracy definition
    y_pred_sigmoid = tf.sigmoid(y_pred)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.greater(y_pred_sigmoid, 0.5), tf.float32), y_target), tf.float32))

    # training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    # specify learning rate
    each_l_rate = 0.005
    # specify weight decay coefficient
    each_W_lambda = 0.0

    train_error_list = []

    # initialize session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(W)
    sess.run(b)

    for step in range(0, 5000):
        batch_idx = step % numBatches
        trainDataBatch = trainData[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)]
        trainTargetBatch = trainTarget[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)]
        _, currentW, currentb, yhat = sess.run([train, W, b, y_pred],
                                               feed_dict={X: trainDataBatch, y_target: trainTargetBatch,
                                                          l_rate: each_l_rate, W_lambda: each_W_lambda})
        if batch_idx == numBatches - 1:
            train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch,
                                                    W_lambda: each_W_lambda})
            train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})

    training_error = sess.run(loss,
                              feed_dict={X: trainDataBatch, y_target: trainTargetBatch, W_lambda: each_W_lambda})
    train_error_list.append(training_error)

    # trainData accuracy
    y = tf.convert_to_tensor(trainTarget)
    y = tf.reshape(y, [3500,-1])
    y = tf.cast(y, tf.float64)
    yhat_norm = tf.matmul(biasTrainData, W_norm)
    yhat_bound_norm = tf.cast(tf.greater(yhat_norm, 0.5), tf.float64)
    equal_norm = tf.cast(tf.equal(y, yhat_bound_norm), tf.float64)
    accuracy_norm = tf.reduce_sum(tf.reduce_sum(equal_norm, 0), 0)/3500
    print("TrainData Accuracy (linear regression normal equation):", sess.run(accuracy_norm))
    accuracy_SGD = sess.run(accuracy, feed_dict={X: trainData, y_target: trainTarget})
    print("TrainData Accuracy (logistic regression SGD):", accuracy_SGD)

    # validData accuracy
    y = tf.convert_to_tensor(validTarget)
    y = tf.reshape(y, [100,-1])
    y = tf.cast(y, tf.float64)
    yhat_norm = tf.matmul(biasValidData, W_norm)
    yhat_bound_norm = tf.cast(tf.greater(yhat_norm, 0.5), tf.float64)
    equal_norm = tf.cast(tf.equal(y, yhat_bound_norm), tf.float64)
    accuracy_norm = tf.reduce_sum(tf.reduce_sum(equal_norm, 0), 0)/100
    print("ValidData Accuracy (linear regression normal equation):", sess.run(accuracy_norm))
    accuracy_SGD = sess.run(accuracy, feed_dict={X: validData, y_target: validTarget})
    print("ValidData Accuracy (logistic regression SGD):", accuracy_SGD)

    # testData accuracy
    y = tf.convert_to_tensor(testTarget)
    y = tf.reshape(y, [145,-1])
    y = tf.cast(y, tf.float64)
    yhat_norm = tf.matmul(biasTestData, W_norm)
    yhat_bound_norm = tf.cast(tf.greater(yhat_norm, 0.5), tf.float64)
    equal_norm = tf.cast(tf.equal(y, yhat_bound_norm), tf.float64)
    accuracy_norm = tf.reduce_sum(tf.reduce_sum(equal_norm, 0), 0)/145
    print("TestData Accuracy (linear regression normal equation):", sess.run(accuracy_norm))
    accuracy_SGD = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget})
    print("TestData Accuracy (logistic regression SGD):", accuracy_SGD)


    # specify learning rate
    each_l_rate = 0.001
    # specify weight decay coefficient
    each_W_lambda = 0.0
    train_error_list = []
    train_accuracy_list = []
    linear_train_error_list = []
    linear_train_accuracy_list = []

    # variable creation
    linear_W = tf.Variable(tf.random_normal(shape=[int(np.shape(trainData)[1]), 1], stddev=0.5, seed=521), name="weights")
    # linear_W = tf.Variable(tf.truncated_normal(shape=[int(np.shape(trainData)[1]), 1], stddev=0.1), name="weights")
    # linear_W = tf.Variable(tf.zeros([trainData.shape[1], 1]), dtype=tf.float32)
    linear_b = tf.Variable(0.0, name="biases")
    linear_X = tf.placeholder(tf.float32, [batch_size, int(np.shape(trainData)[1])], name="input_x")
    linear_y_target = tf.placeholder(tf.float32, [batch_size, 1], name="target_y")

    # graph definition
    linear_y_pred = tf.matmul(linear_X,linear_W) + linear_b

    linear_l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    linear_W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    linear_mse = tf.reduce_sum(tf.square(linear_y_pred-linear_y_target), name="mse")/(2.0*batch_size)
    linear_W_decay = tf.reduce_sum(tf.multiply(linear_W,linear_W))*linear_W_lambda/2
    linear_loss = linear_mse + linear_W_decay

    # training mechanism
    linear_optimizer = tf.train.AdamOptimizer(learning_rate=linear_l_rate)
    linear_train = linear_optimizer.minimize(loss=linear_loss)
    linear_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.greater(linear_y_pred, 0.5), tf.float32), linear_y_target), tf.float32))

    # initialize session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(W)
    sess.run(b)
    sess.run(linear_W)
    sess.run(linear_b)

    for step in range(0, 5000):
        batch_idx = step % numBatches
        trainDataBatch = trainData[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)]
        trainTargetBatch = trainTarget[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)]
        _, currentW, currentb, yhat = sess.run([train, W, b, y_pred],
                                               feed_dict={X: trainDataBatch, y_target: trainTargetBatch,
                                                          l_rate: each_l_rate, W_lambda: each_W_lambda})
        _, currentW, currentb, yhat = sess.run([linear_train, linear_W, linear_b, linear_y_pred],
                                                    feed_dict={linear_X: trainDataBatch, linear_y_target: trainTargetBatch,
                                                               linear_l_rate: each_l_rate, linear_W_lambda: each_W_lambda})
        if batch_idx == numBatches - 1:
            train_error = sess.run(loss, feed_dict={X: trainDataBatch, y_target: trainTargetBatch,
                                                    W_lambda: each_W_lambda})
            linear_train_error = sess.run(linear_loss, feed_dict={linear_X: trainDataBatch, linear_y_target: trainTargetBatch,
                                                                  linear_W_lambda: each_W_lambda})
            train_accuracy = sess.run(accuracy, feed_dict={X: trainDataBatch, y_target: trainTargetBatch})
            linear_train_accuracy = sess.run(linear_accuracy, feed_dict={linear_X: trainDataBatch, linear_y_target: trainTargetBatch})
            train_error_list.append(train_error)
            train_accuracy_list.append(train_accuracy)
            linear_train_error_list.append(linear_train_error)
            linear_train_accuracy_list.append(linear_train_accuracy)

    print("Logistic Regression Cross-Entropy Loss:", train_error_list[-1])
    print("Linear Regression MSE Loss:", linear_train_error_list[-1])
    print("Logistic Regression Accuracy:", train_accuracy_list[-1])
    print("Linear Regression Accuracy:", linear_train_accuracy_list[-1])

    plt.clf()
    plt.plot(train_error_list)
    plt.title('Training Cross-Entropy Loss - Logistic Regression')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Cross-Entropy Loss')
    x1, x2, x3, x4 = plt.axis()
    plt.axis((x1, x2, 0, 0.5))
    plt.savefig('logistic_regression_error.png')
    plt.clf()
    plt.plot(linear_train_error_list)
    plt.title('Training MSE Loss - Linear Regression')
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE Loss')
    x1, x2, x3, x4 = plt.axis()
    plt.axis((x1, x2, 0, 2))
    plt.savefig('linear_regression_loss.png')
    plt.clf()
    plt.plot(train_accuracy_list)
    plt.title('Training Accuracy - Logistic Regression')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('logistic_regression_accuracy.png')
    plt.clf()
    plt.plot(linear_train_accuracy_list)
    plt.title('Training Accuracy - Linear Regression')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('linear_regression_accuracy.png')


    # Comparison between cross-entropy and MSE loss over y = [0,1]
    y = 0
    yhat = np.linspace(0,1,500)
    cross_entropy_loss = -1 * y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
    mse_loss = np.power(y - yhat, 2)
    plt.clf()
    plt.plot(yhat, cross_entropy_loss, label='cross-entropy loss')
    plt.plot(yhat, mse_loss, label='MSE loss')
    plt.title('Cross-Entropy Loss vs MSE loss')
    plt.xlabel('prediction yhat (given y = 0)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cross_entropy_vs_mse_loss.png')

if __name__ == '__main__':
    # q2part1()
    q2part2()
    # q2part3()
    # optimal_l_rate()

