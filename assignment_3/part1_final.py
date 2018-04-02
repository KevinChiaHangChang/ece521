import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

hidden_l_1 = 1000
output_l = 10

batch_size = 500

num_iterations = 1000

def load_data():
	with np.load("notMNIST.npz") as data:
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

def output_to_input(output_x,num_hidden):

	output_x_shape = output_x.get_shape().as_list()
	print("Output x shape: %d" % (output_x_shape[1]))
	
	# perform Xavier initialization
	W = tf.Variable(tf.truncated_normal(shape=[output_x_shape[1],num_hidden], mean=0.0, stddev=3.0/(28*28+num_hidden), dtype=tf.float32), name="W")
	b = tf.Variable(tf.zeros(shape=[num_hidden]), dtype=tf.float32, name="b")

	# # combine weights and bias
	# W = tf.concat([b,W],0)

	# # add 1 to output x
	# ones = tf.ones([output_x.get_shape().as_list()[0],1], dtype=tf.float32)
	# output_x = tf.concat([ones,output_x],1)

	# compute weight input s
	s = tf.add(tf.matmul(output_x,W),b)

	# DEBUGGING
	# input_s_shape = s.get_shape().as_list()
	# print("Input s shape: %d" % (input_s_shape[1]))

	return s,W

def reshape_target(target,num):
	x = np.zeros((len(target), num))
	x[np.arange(len(target)), target] = 1
	return x

if __name__ == '__main__':

	# load data
	trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

	epoch_size = len(trainData)
	valid_size = len(validData)
	test_size = len(testData)
	numBatches = epoch_size/batch_size
	print("Number of batches: %d " % numBatches)

	# flatten training data
	trainData = np.reshape(trainData,[epoch_size,-1])
	print("Train data size: %d x %d" % (trainData.shape[0],trainData.shape[1]))
	# flatten validation data
	validData = np.reshape(validData,[valid_size,-1])
	print("Validation data size: %d x %d" % (validData.shape[0],validData.shape[1]))
	# flatten test data
	testData = np.reshape(testData,[test_size,-1])
	print("Test data size: %d x %d" % (testData.shape[0],testData.shape[1]))

	# reshape training target
	trainTarget = reshape_target(trainTarget,10)
	print("Train target size: %d x %d" % (trainTarget.shape[0],trainTarget.shape[1]))
	# flatten validation target
	validTarget = reshape_target(validTarget,10)
	print("Validation target size: %d x %d" % (validTarget.shape[0],validTarget.shape[1]))
	# faltten test target
	testTarget = reshape_target(testTarget,10)
	print("Test target size: %d x %d" % (testTarget.shape[0],testTarget.shape[1]))


	# define model
	x_in = tf.placeholder(shape=[None,28*28], dtype=tf.float32, name="x_in")
	y_target = tf.placeholder(dtype=tf.float32, name="y_target")

	# neural network layers
	s_in1,W_1 = output_to_input(x_in,hidden_l_1)
	x_out1 = tf.nn.relu(s_in1)

	s_in2,W_2 = output_to_input(x_out1,output_l)
	# DEBUGGING
	# print("Input s shape: %d" % (s_in2.get_shape().as_list()[1]))



	# loss function
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_in2, labels=y_target)/2.0)

	# accuracy
	y_pred_softmax = tf.nn.softmax(s_in2)
	correct_predictions = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(y_target, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), 0)
	incorrect_predictions = tf.not_equal(tf.argmax(y_pred_softmax, 1), tf.argmax(y_target, 1))
	classification_error = tf.reduce_mean(tf.cast(incorrect_predictions, tf.float32), 0)

	# training
	l_rate = tf.placeholder(dtype=tf.float32, name="l_rate")
	train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss=cross_entropy_loss)



	# learning rates
	# BEST learning rate: 0.005
	learning_rates = [0.005,0.001,0.0001]

	train_error_list = []
	train_classification_list = []
	valid_error_list = []
	valid_classification_list = []
	test_error_list = []
	test_classification_list = []

	# main loop
	for each_learning_rate in learning_rates:

		print("Learning rate: %f \n" % each_learning_rate)

		# initialize variables
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		for step in range(0,num_iterations):

			batch_idx = step%numBatches
			trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
			trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]

			sess.run([train,W_1,W_2], feed_dict={x_in: trainDataBatch, y_target: trainTargetBatch, l_rate: each_learning_rate})
			if batch_idx == numBatches-1:
				curr_train_error,curr_train_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: trainDataBatch, y_target: trainTargetBatch})
				curr_valid_error,curr_valid_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: validData, y_target: validTarget})
				curr_test_error,curr_test_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: testData, y_target: testTarget})
				# print("Training error: %f" % curr_train_error)
				# print("Training accuracy: %f" % curr_train_accuracy)
				# print("Validation error: %f" % curr_valid_error)
				# print("Validation accuracy: %f" % curr_valid_accuracy)
				# print("Test error: %f" % curr_test_error)
				# print("Test accuracy: %f" % curr_test_accuracy)
				test_error_list.append(curr_train_error)
				test_classification_list.append(curr_train_classification)
				valid_error_list.append(curr_valid_error)
				valid_classification_list.append(curr_valid_classification)
				test_error_list.append(curr_test_error)
				test_classification_list.append(curr_test_classification)

			if step%100 == 0:
				print("Step: %d " % step)

	# plot image
	plt.clf()
	plt.plot(train_error_list, label="training cross entropy loss")
	plt.plot(valid_error_list, label="validation cross entropy loss")
	# plt.plot(test_error_list, label="test cross entropy loss")
	plt.title("cross entropy loss for 1-layer neural network")
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("cross entropy loss")
	plt.savefig("1_1_2_cross_entropy_loss.png")

	plt.clf()
	plt.plot(train_classification_list, label="training classification error")
	plt.plot(valid_classification_list, label="validation classification error")
	# plt.plot(test_classification_list, label="test classification error")
	plt.title("classification error for 1-layer neural network")
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("classification error")
	plt.savefig("1_1_2_classification_error.png")





