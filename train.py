import tensorlayer as tl
import tensorflow as tf
from dataset import *
from model import *
import sys

lfw_Path = './lfw/LFW-gender-folds.txt'
test_Path = './face/face_list.txt'

def train():
	# Learning params
	learning_rate = 0.0005
	training_iters = 100
	batch_size = 50
	display_step = 1
	test_step = 1
	f = open('./test_result.txt','w')

	# Network params
	image_size = (224, 224)
	n_classes = 2

	#sess = tf.InteractiveSession()

	x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
	y = tf.placeholder(tf.float32, [batch_size, n_classes])

	def training(sess, network, inputs, labels):
		feed_dict={x:inputs, y:labels}
		feed_dict.update( network.all_drop )
		batch_loss,_ = sess.run([loss, optimizer], feed_dict=feed_dict)
		return batch_loss

	def comput_acc(sess, network, inputs, labels):
		dp_dict = {x: 1 for x in network.all_drop}
		feed_dict = {x: inputs, y: labels}
		feed_dict.update(dp_dict)
		acc = sess.run(accuracy, feed_dict=feed_dict)
		return acc

	network = vggnet(x)
	pred = network.outputs
	#print(y.get_shape().ndims, y_.get_shape().ndims)
	
	# Loss and optimizer
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# Load dataset
	data = Dataset(lfw_Path)
	
	#config = tf.ConfigProto() 
	#config.gpu_options.allow_growth = True
	with tf.Session() as sess:
		print 'Init variable'
		f.write('Init \n')
		sess.run(tf.initialize_all_variables())
		
		load_npy('./vggface_100.npy', sess, '')
		
		print 'Start training'
		f.write('Start training \n')

		step = 1
		while step < training_iters:
			batch_train_images, batch_train_labels = data.next_batch(batch_size,'train')
			batch_loss = training(sess, network, batch_train_images, batch_train_labels)

			# Display testing status
			if step % test_step == 0:
				test_acc = 0
				test_count = 0
				print 'test:'
				f.write('test \n')
				for i in range(int(data.test_size/batch_size)):
					batch_test_images, batch_test_labels = data.next_batch(batch_size,'test')
					acc = comput_acc(sess, network, batch_test_images, batch_test_labels)
					test_acc += acc
					test_count += 1
				test_acc /= test_count
				print ('test_accuracy: ', test_acc)
				f.write('test_accuracy: ' + str(test_acc)+'\n')
				print 'finish test'
				f.write('finish test \n')

			# Display training status
			if step % display_step == 0:
				print 'display:'
				f.write('display \n')
				acc = comput_acc(sess, network, batch_train_images, batch_train_labels)
				print ('loss: ', batch_loss, ' accuracy: ' , acc)			
				f.write('loss: '+str(batch_loss)+' accuracy: '+str(acc) + '\n')
				print 'finish display'
				f.write('finish display \n')
			step += 1

		save_npy(sess, 'vggface_100.npy')
		print "Finish!"
		f.write('Finish! \n')

def test(malef, femalef):
	# testing params
	batch_size = 1
	display_step = 1
	test_step = 1

	# Network params
	image_size = (224, 224)
	n_classes = 2

	x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

	network = vggnet(x)
	predict = network.outputs
	prob = tf.nn.softmax(predict)

	def testing(sess, network, inputs):
		feed_dict={x:inputs}
		dp_dict = {x: 1 for x in network.all_drop}
		feed_dict.update( dp_dict )
		pred = sess.run([prob], feed_dict=feed_dict)
		return pred

	# Load dataset
	data = Testset(test_Path)

	#config = tf.ConfigProto() 
	#config.gpu_options.allow_growth = True
	with tf.Session() as sess:
		print 'Init variable'
		sess.run(tf.initialize_all_variables())
		
		load_npy('./vgg_face_50.npy', sess, '')
		
		print 'Start training'

		step = 1
		while step < data.test_size:
			batch_test_images, image_name = data.next_batch(batch_size)
			prediction = testing(sess, network, batch_test_images)
			if prediction[0][0][0] > prediction[0][0][1]:
				gender = 'M'
				malef.write(str(image_name) + '\n')
			else:
				gender = 'F'
				femalef.write(str(image_name) + '\n')
			print('pred: ', prediction, gender, image_name)
			step += 1

		print "Finish!"

if __name__ == '__main__':
	#train()
	male_file = open('male.txt', 'w')
	female_file = open('female.txt', 'w')
	test(male_file, female_file)
