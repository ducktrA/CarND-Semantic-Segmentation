import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
       
    
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) # to load the model and weights
    graph = tf.get_default_graph()

    print("Named layers in graph: ", [m.name for m in graph.get_operations()])

    vgg_input_tensor_name = 'image_input:0'

    # dropout
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    inp = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    #return None, None, None, None, None
    return inp, keep, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Encoder - attach encoder to vgg_layer7_out
         
    #print("vgg_layer7_out.shape: ", vgg_layer7_out.get_shape()) # (?,?,?,4096)

    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, name = "conv_1x1", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	
    #print("conv_1x1 shape: ", conv_1x1.get_shape())
    
    transpose_conv_one = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, name = "transpose_conv_one", padding="same", kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    #print("transpose_conv__one shape: ", transpose_conv_one.get_shape())
    #print("vgg_layer4_out shape: ", vgg_layer4_out.get_shape())

    # transpose_conv_one has shape (?,?,?,2), vgg_layer4_out has shape (?,?,?,512)
    # either use output_shape

    vgg_layer4_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #print("vgg_layer4_out shape: ", vgg_layer4_out.get_shape())


    transpose_conv_one = tf.add(transpose_conv_one, vgg_layer4_out)

    transpose_conv_two = tf.layers.conv2d_transpose(transpose_conv_one, num_classes, 4, 2, padding="SAME", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer3_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    transpose_conv_two = tf.add(transpose_conv_two, vgg_layer3_out)

    # where does the 16 and 8 come from? between the image and layer 3 there a two maxpool layers which shrink the resolution two times by two
    transpose_conv_three = tf.layers.conv2d_transpose(transpose_conv_two, num_classes, 16, 8, padding="SAME" , kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return transpose_conv_three

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1,num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label image
sess    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    print("Start training the network")
    for epoch in range(epochs):
    	print("Epoch: ", epoch)
    	for image, label in get_batches_fn(batch_size):

    		feed_dict = {input_image: image , correct_label: label, keep_prob: 0.8, learning_rate: 0.00005}

    		# i missed the cross_entropy loss. the failure message was "numpy.int32 object is not iterable"
    		_, loss = sess.run([train_op, cross_entropy_loss], feed_dict)

    		print("Loss:", loss)

  
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    epochs = 100
    batch_size = 16  # check memory consumption on gpu
	
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        #keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

        # TODO: Build NN using load_vgg, layers, and optimize function
        inp, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        segmented = layers(layer3, layer4, layer7, num_classes)

        logits, train_op, cross_entropy_loss = optimize(segmented, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function

        # dropout active during training
    	# ok, graph is completely defined, run the initializer
        sess.run(tf.global_variables_initializer())    

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, inp,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, inp)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
