import tensorflow as tf
import numpy as np
import tensorlayer as tl

def load_npy(data_path, session, skip_layer):
    """
    Load model

    Parameters
    ----------
    data_path  : the model's path
    session    : the tensorflow session
    skip_layer : skip the model parameters which you want to skip

    Examples
    -------- 
    load_npy('./vggface_100.npy', session, 'fc8')
    """
    data_dict = np.load(data_path).item()
    for key in data_dict:
        print(key)
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key].values()):
                    session.run(tf.get_variable(subkey).assign(data))

layers_names =['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']

def save_npy(sess, name):
    """
    Save model

    Parameters
    ----------
    sess : the tensorflow session
    name : the model name 

    Examples
    -------- 
    save_npy(session, 'vggface_100.npy')
    """
    save_dict = {}
    for key in layers_names:
        with tf.variable_scope(key, reuse=True):
            weights = tf.get_variable('weights').eval(sess)
            biases = tf.get_variable('biases').eval(sess)
            dp_dict = {key: {'weights':weights, 'biases':biases}}
            save_dict.update(dp_dict)
    tl.files.save_any_to_npy(save_dict=save_dict,name=name)

def vggnet(_X):
    """
    the vggnet model

    Parameters
    ----------
    _X : input pictures
    """
    network = tl.layers.InputLayer(_X, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool5')
    
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc6')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc7')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=2,
                        act = tf.identity,
                        name = 'fc8')
    
    return network  
