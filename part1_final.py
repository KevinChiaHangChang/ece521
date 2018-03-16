import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def timestamp(message, last_time=0):

    if last_time ==0:
        elapsed_time = timeit.default_timer()
    else:
        elapsed_time = timeit.default_timer()-last_time
        print(message, elapsed_time)
    return elapsed_time

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

def q1part1():

    # load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = data_gen()
    trainData = np.reshape(trainData,[3500,-1])

    sess = tf.InteractiveSession()

    batch_size = 500
    numBatches = len(trainData)/batch_size
    print("Number of batches: %d" % numBatches)

    # variable creation
    W = tf.Variable(tf.truncated_normal(shape=[int(np.shape(trainData)[1]), 1], stddev=0.5), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, [batch_size, int(np.shape(trainData)[1])], name="input_x")
    y_target = tf.placeholder(tf.float32, [batch_size, 1], name="target_y")

    # graph definition
    y_pred = tf.matmul(X,W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    mse = tf.reduce_mean(tf.square(y_pred-y_target)/2.0, name="mse")
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2
    loss = mse + W_decay

    # training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    error = list()
    l_rates = [0.005,0.001,0.0001]

    for count, each_l_rate in enumerate(l_rates):
        # initialize session
        init = tf.global_variables_initializer()

        sess = tf.InteractiveSession()
        sess.run(init)

        initialW = sess.run(W)
        initialb = sess.run(b)

        print("Learning rate: %f " % each_l_rate)
        err_list = []
        minimal_error = 100
        for step in range(0,20000):
            batch_idx = step%numBatches
            trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            _, err, currentW, currentb, yhat = sess.run([train, loss, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: 0.0})
            if batch_idx == numBatches-1:
                err_list.append(err)
                # print("Step: %d " % step)
                # print("Error: %f" % err)
            minimal_error = min(minimal_error,err)
            # if step%1000 == 0:
            #     print("Step: %d " % step)
            #     print("Error: %f" % err)
        print("MSE error is : %f" % err)
        error.append(err_list)
    # plot image
    plt.clf()
    plt.plot(error[0], label='learning rate: 0.005')
    plt.plot(error[1], label='learning rate: 0.001')
    plt.plot(error[2], label='learning rate: 0.0001')
    x1, x2, x3, x4 = plt.axis()
    plt.axis((x1, x2, 0, 10))
    plt.title("linear regression w/ stochastic gradient descent")
    plt.legend()
    plt.xlabel('number of epoch')
    plt.ylabel('MSE error')
    plt.savefig("sgd_learning_rate.png")

def q1part2():

    # load data
    realTrainData, realTrainTarget, validData, validTarget, testData, testTarget = data_gen()
    realTrainData = np.reshape(realTrainData,[3500,-1])

    sess = tf.InteractiveSession()

    batch_list = [500, 1500, 3500]
    error = list()

    for batch_size in batch_list:

        if batch_size == 1500:
            trainData = np.concatenate((realTrainData[:], realTrainData[:1000]))
            trainTarget = np.concatenate((realTrainTarget[:], realTrainTarget[:1000]))
        else:
            trainData = realTrainData
            trainTarget = realTrainTarget

        numBatches = math.ceil(len(trainData)/batch_size)
        print("Number of batches: %d" % numBatches)

        # variable creation
        W = tf.Variable(tf.truncated_normal(shape=[int(np.shape(trainData)[1]), 1], stddev=0.5), name="weights")
        b = tf.Variable(0.0, name="biases")
        X = tf.placeholder(tf.float32, [batch_size, int(np.shape(trainData)[1])], name="input_x")
        y_target = tf.placeholder(tf.float32, [batch_size, 1], name="target_y")

        # graph definition
        y_pred = tf.matmul(X,W) + b

        # need to define W_lambda, l_rate
        l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

        # error definition
        mse = tf.reduce_mean(tf.square(y_pred-y_target)/2.0, name="mse")
        W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2
        loss = mse + W_decay

        # training mechanism
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
        train = optimizer.minimize(loss=loss)

        each_l_rate = 0.005

        # initialize session
        init = tf.global_variables_initializer()

        sess = tf.InteractiveSession()
        sess.run(init)

        initialW = sess.run(W)
        initialb = sess.run(b)

        print("Batch size: %f " % batch_size)
        err_list = []
        minimal_error = 100
        index = 0
        for step in range(0,20000):
            batch_idx = step%numBatches
            trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            index = (index + batch_size) % 3500
            _, err, currentW, currentb, yhat = sess.run([train, loss, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: 0.0})
            if batch_idx == numBatches-1:
                err_list.append(err)
            minimal_error = min(minimal_error,err)
            # if step%5000 == 0:
            #     print("Step: %d " % step)
            #     print("Error: %f" % err)
        error.append(err_list)
        print("The MSE error is: %f" % err)

    # plot image
    plt.clf()
    plt.plot(error[0], label='batch size: 500')
    plt.plot(error[1], label='batch size: 1500')
    plt.plot(error[2], label='batch size: 3500')
    x1, x2, x3, x4 = plt.axis()
    plt.axis((x1, x2, 0, 1))
    plt.title("linear regression w/ stochastic gradient descent")
    plt.legend()
    plt.xlabel('number of epoch')
    plt.ylabel('MSE error')
    plt.savefig("sgd_batch_size.png")

def q1part3():
    # load data
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
    # DEBUGGING
    # print(np.shape(trainData))
    # print(np.shape(trainTarget))

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
    # DEBUGGING
    # print(sess.run(tf.shape(trainDataTensor)))

    # variable creation
    W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 1], stddev=0.5), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, name="input_x")
    y_target = tf.placeholder(tf.float32, name="target_y")
    # DEBUGGING
    # print(sess.run(tf.shape(W)))
    # print(sess.run(tf.shape(y_target)))

    # graph definition
    y_pred = tf.matmul(X,W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    mse = tf.reduce_mean(tf.square(y_pred-y_target)/2.0, name="mse")
    # DEBUGGING (NORM might not be right function to use)
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2
    loss = mse + W_decay

    # accuracy definition
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred,y_target), tf.float32))

    # training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    # initialize W_lambdas
    accuracy_list = []
    weight_list = []
    W_lambdas = [0.0, 0.001, 0.1, 1]
    for count, each_W_lambda in enumerate(W_lambdas):

        # initialize session
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # initialize weights & bias
        initialW = sess.run(W)
        initialb = sess.run(b)

        # print("Learning rate: %f " % each_l_rate)
        print("Weight decay coefficient: %f " % each_W_lambda)

        # initialize error list
        err_list = []
        # initialize minimum error
        minimal_error = 100

        trainDataRand, trainTargetRand = trainData, trainTarget

        for step in range(0,20000):

            # compute batch index
            batch_idx = step%numBatches

            # randomize
            if batch_idx == 0:
                randIdx = np.arange(epoch_size)
                np.random.shuffle(randIdx)
                trainDataRand, trainTargetRand = trainData[randIdx[:epoch_size]], trainTarget[randIdx[:epoch_size]]

            # extract training data batch
            trainDataBatch = trainDataRand[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            # extract training target batch
            trainTargetBatch = trainTargetRand[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
            # train
            _, err, currentW, currentb, yhat = sess.run([train, loss, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: 0.005, W_lambda: each_W_lambda})
            # append error when reach maximum batch index
            if batch_idx == numBatches-1:
                err_list.append(err)
            # update minimum error
            minimal_error = min(minimal_error,err)

            # DEBUGGING
            if step%5000 == 0:
                print("Step: %d " % step)
                print("Error: %f " % err)


        W_SGD = tf.cast(tf.convert_to_tensor(currentW), tf.float64)
        b_SGD = tf.cast(tf.convert_to_tensor(currentb), tf.float64)

        validationData = tf.convert_to_tensor(validData)
        validationData = tf.reshape(validationData, [100, -1])

        y = tf.convert_to_tensor(validTarget)
        y = tf.reshape(y, [100,-1])
        y = tf.cast(y, tf.float64)

        yhat_SGD = tf.add(tf.matmul(validationData, W_SGD), b_SGD)
        yhat_bound_SGD = tf.cast(tf.greater(yhat_SGD, 0.5), tf.float64)
        mse_SGD = tf.reduce_sum(tf.square(yhat_SGD-y), name="mse_SGD")/(2.0*100)
        equal_SGD = tf.cast(tf.equal(y, yhat_bound_SGD), tf.float64)
        accuracy_SGD = tf.reduce_sum(tf.reduce_sum(equal_SGD, 0), 0)/100

        # print(sess.run(yhat_SGD))

        print(sess.run(mse_SGD))
        print(sess.run(accuracy_SGD))

        accuracy_list.append(sess.run(accuracy_SGD))
        weight_list.append((W_SGD, b_SGD))


    best_lambda = W_lambdas[accuracy_list.index(max(accuracy_list))]
    best_weight, best_bias = weight_list[accuracy_list.index(max(accuracy_list))]
    print (best_lambda)

    testData = tf.convert_to_tensor(testData)
    testData = tf.reshape(testData, [145, -1])

    y = tf.convert_to_tensor(testTarget)
    y = tf.reshape(y, [145,-1])
    y = tf.cast(y, tf.float64)

    yhat_SGD = tf.add(tf.matmul(testData, best_weight), best_bias)
    yhat_bound_SGD = tf.cast(tf.greater(yhat_SGD, 0.5), tf.float64)
    mse_SGD = tf.reduce_sum(tf.square(yhat_SGD-y), name="mse_SGD")/(2.0*145)
    equal_SGD = tf.cast(tf.equal(y, yhat_bound_SGD), tf.float64)
    accuracy_SGD = tf.reduce_sum(tf.reduce_sum(equal_SGD, 0), 0)/145

    print ("Test data 's accuracy %f with lambda %f" %(sess.run(accuracy_SGD), best_lambda))


    # # validation set accuracy
    # # err_validation = sess.run(loss, feed_dict={X: validData, y_target: validTarget, l_rate: 0.005, W_lambda: each_W_lambda})
    # accuracy_validation, y_hathat = sess.run([accuracy, y_pred], feed_dict={X: validData, y_target: validTarget, l_rate: 0.005, W_lambda: each_W_lambda})
    # # print("Validation error: %f " % err_validation)
    #
    #
    # # print(validTarget)
    # print(validTarget)
    #
    # print("Validation accuracy: %f " % accuracy_validation)
    #
    # # test set accuracy
    # # err_test = sess.run(loss, feed_dict={X: testData, y_target: testTarget, l_rate: 0.005, W_lambda: each_W_lambda})
    # accuracy_test = sess.run(accuracy, feed_dict={X: testData, y_target: testTarget, l_rate: 0.005, W_lambda: each_W_lambda})
    # # print("Test error: %f " % err_test)
    # print("Test accuracy: %f " % accuracy_test)

def q1part4():

    sess = tf.InteractiveSession()

    trainData, trainTarget, validData, validTarget, testData, testTarget = data_gen()

    elapsed_time_norm = timestamp("")
    realTrainData = tf.convert_to_tensor(trainData)
    realTrainData = tf.reshape(realTrainData, [3500, -1])
    biasVec = tf.convert_to_tensor([1 for i in range(3500)])
    biasVec = tf.cast(biasVec, tf.float64)
    biasVec = tf.reshape(biasVec, [-1, 3500])
    realTrainData = tf.transpose(realTrainData)
    realTrainData = tf.concat([biasVec, realTrainData], 0)
    realTrainData = tf.transpose(realTrainData)
    realTrainTarget = tf.convert_to_tensor(trainTarget)
    realTrainTarget = tf.cast(realTrainTarget, tf.float64)

    Wnorm = tf.matmul(tf.transpose(realTrainData), realTrainData)
    Wnorm = tf.matrix_inverse(Wnorm)
    Wnorm = tf.matmul(Wnorm, tf.transpose(realTrainData))
    Wnorm = tf.matmul(Wnorm, realTrainTarget)

    elapsed_time_norm = timestamp("Computation time for W using normal equation:", elapsed_time_norm)

    trainData = np.reshape(trainData,[3500,-1])

    batch_size = 500
    numBatches = len(trainData)/batch_size

    # variable creation
    W = tf.Variable(tf.truncated_normal(shape=[int(np.shape(trainData)[1]), 1], stddev=0.5), name="weights")
    b = tf.Variable(0.0, name="biases")
    X = tf.placeholder(tf.float32, [batch_size, int(np.shape(trainData)[1])], name="input_x")
    y_target = tf.placeholder(tf.float32, [batch_size, 1], name="target_y")

    # graph definition
    y_pred = tf.matmul(X,W) + b

    # need to define W_lambda, l_rate
    l_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    W_lambda = tf.placeholder(tf.float32, [], name="weight_decay")

    # error definition
    mse = tf.reduce_mean(tf.square(y_pred-y_target)/2.0, name="mse")
    W_decay = tf.reduce_sum(tf.multiply(W,W))*W_lambda/2
    loss = mse + W_decay

    # training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss=loss)

    each_l_rate = 0.005
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    elapsed_time_SGD = timestamp("")
    for step in range(0,20000):
        batch_idx = step%numBatches
        trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
        _, err, currentW, currentb, yhat = sess.run([train, loss, W, b, y_pred], feed_dict={X: trainDataBatch, y_target: trainTargetBatch, l_rate: each_l_rate, W_lambda: 0.0})

    elapsed_time_SGD = timestamp("Computation time for W using SGD:", elapsed_time_SGD)
    W_SGD = tf.cast(tf.convert_to_tensor(currentW), tf.float64)
    b_SGD = tf.cast(tf.convert_to_tensor(currentb), tf.float64)

    realTestData = tf.convert_to_tensor(testData)
    realTestData = tf.reshape(realTestData, [145, -1])
    biasVec = tf.convert_to_tensor([1 for i in range(145)])
    biasVec = tf.cast(biasVec, tf.float64)
    biasVec = tf.reshape(biasVec, [-1, 145])
    realTestData = tf.transpose(realTestData)
    realTestData = tf.concat([biasVec, realTestData], 0)
    realTestData = tf.transpose(realTestData)

    testData = tf.convert_to_tensor(testData)
    testData = tf.reshape(testData, [145, -1])

    y = tf.convert_to_tensor(testTarget)
    y = tf.reshape(y, [145,-1])
    y = tf.cast(y, tf.float64)

    yhat_norm = tf.matmul(realTestData, Wnorm)
    yhat_bound_norm = tf.cast(tf.greater(yhat_norm, 0.5), tf.float64)
    mse_norm = tf.reduce_sum(tf.square(yhat_norm-y), name="mse_norm")/(2.0*145)
    equal_norm = tf.cast(tf.equal(y, yhat_bound_norm), tf.float64)
    accuracy_norm = tf.reduce_sum(tf.reduce_sum(equal_norm, 0), 0)/145

    # print(sess.run(yhat_norm))

    print(sess.run(mse_norm))
    print(sess.run(accuracy_norm))

    yhat_SGD = tf.add(tf.matmul(testData, W_SGD), b_SGD)
    yhat_bound_SGD = tf.cast(tf.greater(yhat_SGD, 0.5), tf.float64)
    mse_SGD = tf.reduce_sum(tf.square(yhat_SGD-y), name="mse_SGD")/(2.0*145)
    equal_SGD = tf.cast(tf.equal(y, yhat_bound_SGD), tf.float64)
    accuracy_SGD = tf.reduce_sum(tf.reduce_sum(equal_SGD, 0), 0)/145

    # print(sess.run(yhat_SGD))

    print(sess.run(mse_SGD))
    print(sess.run(accuracy_SGD))

if __name__ == '__main__':
    # q1part1()
    # q1part2()
    q1part3()
    # q1part4()
