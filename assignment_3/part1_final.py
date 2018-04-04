# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# hidden_l_1 = 1000
# output_l = 10

# batch_size = 100

# num_iterations = 5000

# W_count = 1

# def load_data():
# 	with np.load("notMNIST.npz") as data:
# 		Data, Target = data ["images"], data["labels"]
# 		np.random.seed(521)
# 		randIndx = np.arange(len(Data))
# 		np.random.shuffle(randIndx)
# 		Data = Data[randIndx]/255.
# 		Target = Target[randIndx]
# 		trainData, trainTarget = Data[:15000], Target[:15000]
# 		validData, validTarget = Data[15000:16000], Target[15000:16000]
# 		testData, testTarget = Data[16000:], Target[16000:]
# 	return trainData, trainTarget, validData, validTarget, testData, testTarget

# def output_to_input(output_x,num_hidden):

# 	output_x_shape = output_x.get_shape().as_list()
# 	print("Output x shape: %d" % (output_x_shape[1]))
	
# 	# perform Xavier initialization
# 	global W_count
# 	W_name = "W_"+str(W_count)
# 	b_name = "b_"+str(W_count)
# 	W_count += 1
# 	# print("W count: %d" % W_count)
# 	# W = tf.Variable(tf.truncated_normal(shape=[output_x_shape[1],num_hidden], mean=0.0, stddev=(3.0/(28*28+num_hidden))**0.5, dtype=tf.float32), name=W_name)
# 	W = tf.get_variable(name=W_name, shape=[output_x_shape[1],num_hidden], initializer=tf.contrib.layers.xavier_initializer())
# 	b = tf.get_variable(name=b_name, shape=[num_hidden], initializer=tf.zeros_initializer())

# 	# # combine weights and bias
# 	# W = tf.concat([b,W],0)

# 	# # add 1 to output x
# 	# ones = tf.ones([output_x.get_shape().as_list()[0],1], dtype=tf.float32)
# 	# output_x = tf.concat([ones,output_x],1)

# 	# compute weight input s
# 	s = tf.add(tf.matmul(output_x,W),b)

# 	# DEBUGGING
# 	# input_s_shape = s.get_shape().as_list()
# 	# print("Input s shape: %d" % (input_s_shape[1]))

# 	return s

# def reshape_target(target,num):
# 	x = np.zeros((len(target), num))
# 	x[np.arange(len(target)), target] = 1
# 	return x

# if __name__ == '__main__':

# 	# load data
# 	trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

# 	epoch_size = len(trainData)
# 	valid_size = len(validData)
# 	test_size = len(testData)
# 	numBatches = epoch_size/batch_size
# 	print("Number of batches: %d " % numBatches)

# 	# flatten training data
# 	trainData = np.reshape(trainData,[epoch_size,-1])
# 	print("Train data size: %d x %d" % (trainData.shape[0],trainData.shape[1]))
# 	# flatten validation data
# 	validData = np.reshape(validData,[valid_size,-1])
# 	print("Validation data size: %d x %d" % (validData.shape[0],validData.shape[1]))
# 	# flatten test data
# 	testData = np.reshape(testData,[test_size,-1])
# 	print("Test data size: %d x %d" % (testData.shape[0],testData.shape[1]))

# 	# reshape training target
# 	trainTarget = reshape_target(trainTarget,10)
# 	print("Train target size: %d x %d" % (trainTarget.shape[0],trainTarget.shape[1]))
# 	# flatten validation target
# 	validTarget = reshape_target(validTarget,10)
# 	print("Validation target size: %d x %d" % (validTarget.shape[0],validTarget.shape[1]))
# 	# faltten test target
# 	testTarget = reshape_target(testTarget,10)
# 	print("Test target size: %d x %d" % (testTarget.shape[0],testTarget.shape[1]))


# 	# define model
# 	x_in = tf.placeholder(shape=[None,28*28], dtype=tf.float32, name="x_in")
# 	y_target = tf.placeholder(dtype=tf.float32, name="y_target")

# 	# neural network layers
# 	s_in1 = output_to_input(x_in,hidden_l_1)
# 	x_out1 = tf.nn.relu(s_in1)

# 	s_in2 = output_to_input(x_out1,output_l)
# 	# DEBUGGING
# 	# print("Input s shape: %d" % (s_in2.get_shape().as_list()[1]))

# 	# weights
# 	# W_1_new = tf.get_default_graph().get_tensor_by_name(name="W_1:0")
# 	# W_2_new = tf.get_default_graph().get_tensor_by_name(name="W_2:0")
	
# 	# assert(W_1_new == W_1)
# 	# assert(W_2_new == W_2)

# 	# learning rate & regularization coefficient
# 	l_rate = tf.placeholder(dtype=tf.float32, name="l_rate")

# 	# loss function
# 	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_in2, labels=y_target)/2.0)

# 	# accuracy
# 	y_pred_softmax = tf.nn.softmax(s_in2)
# 	correct_predictions = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(y_target, 1))
# 	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), 0)
# 	incorrect_predictions = tf.not_equal(tf.argmax(y_pred_softmax, 1), tf.argmax(y_target, 1))
# 	classification_error = tf.reduce_mean(tf.cast(incorrect_predictions, tf.float32), 0)

# 	# training
# 	train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss=cross_entropy_loss)



# 	# learning rates
# 	# BEST learning rate: 0.005
# 	learning_rates = [0.005]
# 	# learning_rates = [0.005,0.001,0.0001]

# 	train_error_list = []
# 	train_classification_list = []
# 	valid_error_list = []
# 	valid_classification_list = []
# 	test_error_list = []
# 	test_classification_list = []

# 	# initialize variables
# 	sess = tf.Session()
# 	init = tf.global_variables_initializer()
# 	sess.run(init)

# 	# # main loop
# 	for each_learning_rate in learning_rates:

# 		print("Learning rate: %f \n" % each_learning_rate)

# 		for step in range(0,num_iterations):

# 			batch_idx = step%numBatches
# 			trainDataBatch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
# 			trainTargetBatch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]

# 			sess.run(train, feed_dict={x_in: trainDataBatch, y_target: trainTargetBatch, l_rate: each_learning_rate})
# 			# print("Current weight 1: %s" % currentW_1.T)
# 			# print("Current weight 2: %s" % currentW_2.T)
# 			if batch_idx == numBatches-1:
# 				curr_train_error,curr_train_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: trainData, y_target: trainTarget})
# 				curr_valid_error,curr_valid_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: validData, y_target: validTarget})
# 				curr_test_error,curr_test_classification = sess.run([cross_entropy_loss,classification_error], feed_dict={x_in: testData, y_target: testTarget})
# 				# print("Training error: %f" % curr_train_error)
# 				# print("Training accuracy: %f" % curr_train_accuracy)
# 				# print("Validation error: %f" % curr_valid_error)
# 				# print("Validation accuracy: %f" % curr_valid_accuracy)
# 				# print("Test error: %f" % curr_test_error)
# 				# print("Test accuracy: %f" % curr_test_accuracy)
# 				train_error_list.append(curr_train_error)
# 				train_classification_list.append(curr_train_classification)
# 				valid_error_list.append(curr_valid_error)
# 				valid_classification_list.append(curr_valid_classification)
# 				test_error_list.append(curr_test_error)
# 				test_classification_list.append(curr_test_classification)

# 			if step%100 == 0:
# 				print("Step: %d " % step)

# 	# plot image
# 	plt.clf()
# 	plt.plot(train_error_list, label="training cross entropy loss")
# 	plt.plot(valid_error_list, label="validation cross entropy loss")
# 	# plt.plot(test_error_list, label="test cross entropy loss")
# 	plt.title("cross entropy loss for 1-layer neural network")
# 	plt.legend()
# 	plt.xlabel("number of epochs")
# 	plt.ylabel("cross entropy loss")
# 	plt.savefig("1_1_2_cross_entropy_loss.png")

# 	plt.clf()
# 	plt.plot(train_classification_list, label="training classification error")
# 	plt.plot(valid_classification_list, label="validation classification error")
# 	# plt.plot(test_classification_list, label="test classification error")
# 	plt.title("classification error for 1-layer neural network")
# 	plt.legend()
# 	plt.xlabel("number of epochs")
# 	plt.ylabel("classification error")
# 	plt.savefig("1_1_2_classification_error.png")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# hyper parameters
num_classes = 10
best_l_rate = 0.001
predefined_w_lambda = 3e-4
default_num_hidden = 1000
default_num_layers = 2

def reshape_target(target,depth):
	x = np.zeros((len(target), depth))
	x[np.arange(len(target)), target] = 1
	return x

# load data first
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

	# reshape data
	trainData = np.reshape(trainData,[len(trainData),-1])
	validData = np.reshape(validData,[len(validData),-1])
	testData = np.reshape(testData,[len(testData),-1])
	print("Train data size:",trainData.shape)
	print("Validation data size:",validData.shape)
	print("Test data size:",testData.shape)

	# change target to one-hot 
	# trainTarget = tf.one_hot(trainTarget, depth=num_classes, dtype=tf.float32, axis=-1)
	# validTarget = tf.one_hot(validTarget, depth=num_classes, dtype=tf.float32, axis=-1)
	# testTarget = tf.one_hot(testTarget, depth=num_classes, dtype=tf.float32, axis=-1)
	trainTarget = reshape_target(trainTarget,num_classes)
	validTarget = reshape_target(validTarget,num_classes)
	testTarget = reshape_target(testTarget,num_classes)
	print("Train target size:",trainTarget.shape)
	print("Validation target size:",validTarget.shape)
	print("Test target size:",testTarget.shape)

def layerwise_build(prev_layer_X,num_hidden):

	num_input = prev_layer_X.get_shape().as_list()[1]

	# initialize weights and bias
	W = tf.get_variable(name="W", shape=[num_input,num_hidden], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=(3.0/(num_input+num_hidden))**0.5))
	b = tf.get_variable(name="b", shape=[1,num_hidden], initializer=tf.zeros_initializer())

	# compute next layer z
	next_layer_z = tf.add(tf.matmul(prev_layer_X,W),b)

	return next_layer_z

def buildGraph(l_rate=best_l_rate,num_layers=default_num_layers,num_hidden=default_num_hidden,w_lambda=predefined_w_lambda,dropout=0,visualize=False):

	# define hyperparameters
	num_iterations = 1200
	batch_size = 500
	num_batches = len(trainData)/batch_size

	tf.reset_default_graph()

	# define placeholders
	X = tf.placeholder(name="X", shape=[None,trainData.shape[1]], dtype=tf.float32)
	y_target = tf.placeholder(name="y_target", shape=[None,num_classes], dtype=tf.float32)

	input_layer = X

	# build neural network
	# NOTE: use variable scope to distinguish between first and second layer weights and bias
	for i in range(1,num_layers+1):
		print("Layer:",i)
		with tf.variable_scope("layer_"+str(i)):
			if i == num_layers:
				print("Number of output units:",num_classes)
				input_layer = layerwise_build(input_layer,num_classes)
				if dropout:
					input_layer = tf.nn.dropout(input_layer,dropout)
			else:
				print("Number of hidden units:",num_hidden)
				input_layer = tf.nn.relu(layerwise_build(input_layer,num_hidden))
				if dropout:
					input_layer = tf.nn.dropout(input_layer,dropout)
	y_pred = tf.nn.softmax(input_layer)

	# loss function
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_layer, labels=y_target))
	total_decay = 0
	for i in range(1,num_layers):
		total_decay += w_lambda*(tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("layer_"+str(i)+"/W:0"))))/2.0
	# w_decay1 = w_lambda*(tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("first_layer/W:0"))))/2.0
	# w_decay2 = w_lambda*(tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("second_layer/W:0"))))/2.0
	loss = cross_entropy_loss+total_decay

	# classification error
	num_correct = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_target,1))
	classification_error = 1-tf.reduce_mean(tf.cast(num_correct, tf.float32))

	# train
	train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

	# initialize saver object
	writer = tf.summary.FileWriter("./weights")
	saver = tf.train.Saver()

	# initialize lists to store validation loss and classification error
	train_loss = []
	train_error = []
	valid_loss = []
	valid_error = []
	test_loss = []
	test_error = []

	# initialize list to store weights at completion intervals
	# weights = []
	# bias = []
	complete_25 = False
	complete_50 = False
	complete_75 = False

	# initialize variables
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# main loop
	for step in range(num_iterations):

		# save variables
		# curr_weights = sess.run(tf.get_default_graph().get_tensor_by_name("second_layer/W:0"))
		# curr_bias = sess.run(tf.get_default_graph().get_tensor_by_name("second_layer/b:0"))
		completed = (step/num_iterations)*100
		if completed >= 25 and not complete_25:
			complete_25 = True 
			# weights.append(curr_weights)
			# bias.append(curr_bias)
			saver.save(sess,"./checkpoints/25_model.ckpt")
		elif completed >= 50 and not complete_50:
			complete_50 = True
			# weights.append(curr_weights)
			# bias.append(curr_bias)
			saver.save(sess,"./checkpoints/50_model.ckpt")
		elif completed >= 75 and not complete_75:
			complete_75 = True
			# weights.append(curr_weights)
			# bias.append(curr_bias)
			saver.save(sess,"./checkpoints/75_model.ckpt")
		elif step >= num_iterations-1:
			# weights.append(curr_weights)
			# bias.append(curr_bias)
			saver.save(sess,"./checkpoints/100_model.ckpt")

		# get train data batch
		batch_idx = step%num_batches
		data_batch = trainData[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]
		target_batch = trainTarget[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)]

		# perform gradient descent
		sess.run(train, feed_dict={X:data_batch, y_target:target_batch})

		# get validation loss and classification error
		if batch_idx == num_batches-1:
			curr_train_loss,curr_train_error = sess.run([loss,classification_error], feed_dict={X:trainData, y_target:trainTarget})
			curr_valid_loss,curr_valid_error = sess.run([loss,classification_error], feed_dict={X:validData, y_target:validTarget})
			curr_test_loss,curr_test_error = sess.run([loss,classification_error], feed_dict={X:testData, y_target:testTarget})

			train_loss.append(curr_train_loss)
			train_error.append(curr_train_error)
			valid_loss.append(curr_valid_loss)
			valid_error.append(curr_valid_error)
			test_loss.append(curr_test_loss)
			test_error.append(curr_test_error)

		if step%100 == 0:
			print("Step",step)

	# print("Weights (at 25\% completion):",weights[0])
	# print("Weights (at 50\% completion):",weights[1])
	# print("Weights (at 75\% completion):",weights[2])
	# print("Weights (at 100\% completion):",weights[3])
	# print("Weights (at 100\% progress)",tf.get_default_graph().get_tensor_by_name("second_layer/W:0").eval())

	# restore saved variables
	if visualize:
		weights = []
		# 25% completion
		saver.restore(sess,"./checkpoints/100_model.ckpt")
		weights.append(sess.run(tf.get_default_graph().get_tensor_by_name("layer_1/W:0")))
		# 50% completion
		saver.restore(sess,"./checkpoints/50_model.ckpt")
		weights.append(sess.run(tf.get_default_graph().get_tensor_by_name("layer_1/W:0")))
		# 75% completion
		saver.restore(sess,"./checkpoints/75_model.ckpt")
		weights.append(sess.run(tf.get_default_graph().get_tensor_by_name("layer_1/W:0")))
		# 100% completion
		saver.restore(sess,"./checkpoints/100_model.ckpt")
		weights.append(sess.run(tf.get_default_graph().get_tensor_by_name("layer_1/W:0")))

		return weights

	# saver.restore(sess,"/75/model.ckpt")
	# print("Weights (at 75\% progress)",tf.get_default_graph().get_tensor_by_name("second_layer/W:0").eval())
	# saver.restore(sess,"/50/model.ckpt")
	# print("Weights (at 50\% progress)",tf.get_default_graph().get_tensor_by_name("second_layer/W:0").eval())
	# saver.restore(sess,"/25/model.ckpt")
	# print("Weights (at 25\% progress)",tf.get_default_graph().get_tensor_by_name("second_layer/W:0").eval())

	return train_loss,train_error,valid_loss,valid_error,test_loss,test_error


def part1_1_1():

	# corresponding training loss
	# [0.089978568, 0.071309395, 0.1180885]
	l_rates = [0.005,0.001,0.0001]
	global_train_loss = []
	global_train_error = []
	global_valid_loss = []
	global_valid_error = []
	global_test_loss = []
	global_test_error = []
	for each_l_rate in l_rates:
		train_loss,train_error,valid_loss,valid_error,test_loss,test_error = buildGraph(l_rate=each_l_rate)
		global_train_loss.append(train_loss[-1])
		global_train_error.append(train_error[-1])
		global_valid_loss.append(valid_loss[-1])
		global_valid_error.append(valid_error[-1])
		global_test_loss.append(test_loss[-1])
		global_test_error.append(test_error[-1])

	print("Global train loss",global_train_loss)
	# min_index = global_train_loss.index(min(global_train_loss))
	# print("Best learning rate:",l_rates[min_index])
	# print("\tTraining cross entropy loss:",global_train_loss[min_index],"\tTraining classification error:",global_train_error[min_index])
	# print("\tValidation cross entropy loss:",global_valid_loss[min_index],"\tValidation classification error:",global_valid_error[min_index])
	# print("\tTest cross entropy loss:",global_test_loss[min_index],"\tTest classification error:",global_test_error[min_index])
	return

def part1_1_2():

	train_loss,train_error,valid_loss,valid_error,test_loss,test_error = buildGraph()

	# plot image
	plt.clf()
	plt.plot(train_loss, label="training cross entropy loss")
	plt.plot(valid_loss, label="validation cross entropy loss")
	plt.plot(test_loss, label="testing cross entropy loss")
	plt.title("cross entropy loss for 1-layer neural network")
	plt.grid()
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("cross entropy loss")
	plt.savefig("1_1_2_cross_entropy_loss.png")

	plt.clf()
	plt.plot(train_error, label="training classification error")
	plt.plot(valid_error, label="validation classification error")
	plt.plot(test_error, label="testing classification error")
	plt.title("classification error for 1-layer neural network")
	plt.grid()
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("classification error")
	plt.savefig("1_1_2_classification_error.png")
	return

def part1_3_1():

	train_loss,train_error,valid_loss,valid_error,test_loss,test_error = buildGraph(w_lambda=0,dropout=0.5)

	# plot image
	plt.clf()
	plt.plot(train_loss, label="training cross entropy loss")
	plt.plot(valid_loss, label="validation cross entropy loss")
	plt.plot(test_loss, label="testing cross entropy loss")
	plt.title("cross entropy loss w/ dropout")
	plt.grid()
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("cross entropy loss")
	plt.savefig("1_3_1_cross_entropy_loss_w_dropout.png")

	plt.clf()
	plt.plot(train_error, label="training classification error")
	plt.plot(valid_error, label="validation classification error")
	plt.plot(test_error, label="testing classification error")
	plt.title("classification error w/ dropout")
	plt.grid()
	plt.legend()
	plt.xlabel("number of epochs")
	plt.ylabel("classification error")
	plt.savefig("1_3_1_classification_error_w_dropout.png")
	return

def part1_3_2():

	# no dropout
	weights = buildGraph(visualize=True)

	for i in range(4):
		print(i)
		plt.figure()
		plt.title("non-dropout weights at "+str((i+1)*25)+"\% completion")
		for j in range(1000):
			plt.subplot(25,40,j+1)
			plt.axis('off')
			plt.tick_params(axis='both',left='off',right='off',top='off',bottom='off',labelleft='off',labelright='off',labeltop='off',labelbottom='off')
			plt.imshow(weights[i][:,j].reshape(28,28), cmap="gray")
		plt.savefig("non_dropout_weights_completed_"+str((i+1)*25)+".png")

	# dropout
	d_weights = buildGraph(w_lambda=0,dropout=0.5,visualize=True)

	for i in range(4):
		print(i)
		plt.figure()
		plt.title("dropout weights at "+str((i+1)*25)+"\% completion")
		for j in range(1000):
			plt.subplot(25,40,j+1)
			plt.axis('off')
			plt.tick_params(axis='both',left='off',right='off',top='off',bottom='off',labelleft='off',labelright='off',labeltop='off',labelbottom='off')
			plt.imshow(d_weights[i][:,j].reshape(28,28), cmap="gray")
		plt.savefig("dropout_weights_completed_"+str((i+1)*25)+".png")

	return

# UNFINISHED
def part1_4_1():

	# define 5 random generator seeds
	seeds = [1000611260,1000303502,1000503154,1000612054]
	for each_seed in seeds:
		np.random.seed(each_seed)
		rand_l_rate = np.exp(3*np.random.rand()-7.5)
		rand_num_layers = np.random.randint(1,6)
		rand_num_hidden = np.random.randint(100,501)
		rand_w_lambda = np.exp(3*np.random.rand()-9)
		rand_dropout = np.random.rand()
		print("Seed:",each_seed,"\n\tRandom learning rate:",rand_l_rate,"\n\tRandom number of layers:",rand_num_layers,"\n\tRandom number of hidden units:",rand_num_hidden,"\n\tRandom weight decay coefficient:",rand_w_lambda,"\n\tRandom dropout:",rand_dropout)

		train_loss,train_error,valid_loss,valid_error,test_loss,test_error = buildGraph(l_rate=rand_l_rate,num_layers=rand_num_layers,num_hidden=rand_num_hidden,w_lambda=rand_w_lambda,dropout=rand_dropout)
		print("Seed:",each_seed)
		lowest_valid_loss = min(valid_loss)
		lowest_test_error = min(test_error)

		print("\tValidation loss:",lowest_valid_loss)
		print("\tTest classification error:",lowest_test_error)
	return

if __name__ == '__main__':
	# part1_1_1()
	# part1_1_2()
	# part1_3_1()
	# part1_3_2()
	part1_4_1()








