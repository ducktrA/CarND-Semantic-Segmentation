# Semantic Segmentation
### Introduction
Goal of this project was to get familiar with FCNs and label the free space of the street.
A VGG acted as encoder and a decoder using transposed convolutions was built.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.

The implementation of the decoder is in line of the well known FCN paper by Shelhamer et al. The questions that paper couldnt answer was if the naming of the downloaded vgg network matches with the layer naming in the paper.
Therefore layers were printed using in vgg_net():

  print("Named layers in graph: ", [m.name for m in graph.get_operations()])

The output is below. Tx for telling which layers to use ;-)

When the entire graph was built i could find the layers in the loaded vgg model.

Next the layers() function was built. A 1x1 conv layer was added as in the tutorial, followed by the transposed layers. It took a couple of attempts to get the shapes matching to create ths skip connection. The first transpose_conv_one layer has shape (?,?,?,2), the vgg_layer4_out has shape (?,?,?,512). The output of the intermediate layers was adapted using the same 1x1 convolution as before. 

##### Training
The batch size was choosen by memory constraints. using nvidia-smi showed that gpu memory is getting tight.

![alt text](https://github.com/ducktrA/CarND-Semantic-Segmentation/blob/master/runs/1505396689.8156643/uu_000089.png?raw=true)
epochs = 100, batch_size = 16, dropout = 0.5, learning_rate = 0.00005
Visually good, few artifacts in top and bottom row. Final mean average loss ~0.16

![alt text](https://github.com/ducktrA/CarND-Semantic-Segmentation/blob/master/runs/1505992214.6061983/uu_000089.png?raw=true)
epochs = 100, batchsize = 16, dropout = 0.8, learning_rage = 0.00005 Visually good, denser labeling, almost no artefacts, final mean average loss ~ 0.07

![alt text](https://github.com/ducktrA/CarND-Semantic-Segmentation/blob/master/runs/1505994691.4699028/uu_000089.png?raw=true)
epochs = 50, batchsize = 16, dropout = 0.8, learning_rage = 0.00005 Visually good, denser labeling, almost no artefacts, final mean average loss ~ 0.66 --> epochs are needed due to fairly low learning rate

##### Network Layers

Named layers in graph:  ['correct_label', 'learning_rate', 'image_input', 'keep_prob', 'Processing/split/split_dim', 'Processing/split', 'Processing/sub/y', 'Processing/sub', 'Processing/sub_1/y', 'Processing/sub_1', 'Processing/sub_2/y', 'Processing/sub_2', 'Processing/concat/axis', 'Processing/concat', 'conv1_1/filter/Initializer/Const', 'conv1_1/filter', 'conv1_1/filter/Assign', 'conv1_1/filter/read', 'conv1_1/L2Loss', 'conv1_1/weight_loss/y', 'conv1_1/weight_loss', 'conv1_1/Conv2D', 'conv1_1/biases/Initializer/Const', 'conv1_1/biases', 'conv1_1/biases/Assign', 'conv1_1/biases/read', 'conv1_1/BiasAdd', 'conv1_1/Relu', 'conv1_2/filter/Initializer/Const', 'conv1_2/filter', 'conv1_2/filter/Assign', 'conv1_2/filter/read', 'conv1_2/L2Loss', 'conv1_2/weight_loss/y', 'conv1_2/weight_loss', 'conv1_2/Conv2D', 'conv1_2/biases/Initializer/Const', 'conv1_2/biases', 'conv1_2/biases/Assign', 'conv1_2/biases/read', 'conv1_2/BiasAdd', 'conv1_2/Relu', 'pool1', 'conv2_1/filter/Initializer/Const', 'conv2_1/filter', 'conv2_1/filter/Assign', 'conv2_1/filter/read', 'conv2_1/L2Loss', 'conv2_1/weight_loss/y', 'conv2_1/weight_loss', 'conv2_1/Conv2D', 'conv2_1/biases/Initializer/Const', 'conv2_1/biases', 'conv2_1/biases/Assign', 'conv2_1/biases/read', 'conv2_1/BiasAdd', 'conv2_1/Relu', 'conv2_2/filter/Initializer/Const', 'conv2_2/filter', 'conv2_2/filter/Assign', 'conv2_2/filter/read', 'conv2_2/L2Loss', 'conv2_2/weight_loss/y', 'conv2_2/weight_loss', 'conv2_2/Conv2D', 'conv2_2/biases/Initializer/Const', 'conv2_2/biases', 'conv2_2/biases/Assign', 'conv2_2/biases/read', 'conv2_2/BiasAdd', 'conv2_2/Relu', 'pool2', 'conv3_1/filter/Initializer/Const', 'conv3_1/filter', 'conv3_1/filter/Assign', 'conv3_1/filter/read', 'conv3_1/L2Loss', 'conv3_1/weight_loss/y', 'conv3_1/weight_loss', 'conv3_1/Conv2D', 'conv3_1/biases/Initializer/Const', 'conv3_1/biases', 'conv3_1/biases/Assign', 'conv3_1/biases/read', 'conv3_1/BiasAdd', 'conv3_1/Relu', 'conv3_2/filter/Initializer/Const', 'conv3_2/filter', 'conv3_2/filter/Assign', 'conv3_2/filter/read', 'conv3_2/L2Loss', 'conv3_2/weight_loss/y', 'conv3_2/weight_loss', 'conv3_2/Conv2D', 'conv3_2/biases/Initializer/Const', 'conv3_2/biases', 'conv3_2/biases/Assign', 'conv3_2/biases/read', 'conv3_2/BiasAdd', 'conv3_2/Relu', 'conv3_3/filter/Initializer/Const', 'conv3_3/filter', 'conv3_3/filter/Assign', 'conv3_3/filter/read', 'conv3_3/L2Loss', 'conv3_3/weight_loss/y', 'conv3_3/weight_loss', 'conv3_3/Conv2D', 'conv3_3/biases/Initializer/Const', 'conv3_3/biases', 'conv3_3/biases/Assign', 'conv3_3/biases/read', 'conv3_3/BiasAdd', 'conv3_3/Relu', 'pool3', 'layer3_out', 'conv4_1/filter/Initializer/Const', 'conv4_1/filter', 'conv4_1/filter/Assign', 'conv4_1/filter/read', 'conv4_1/L2Loss', 'conv4_1/weight_loss/y', 'conv4_1/weight_loss', 'conv4_1/Conv2D', 'conv4_1/biases/Initializer/Const', 'conv4_1/biases', 'conv4_1/biases/Assign', 'conv4_1/biases/read', 'conv4_1/BiasAdd', 'conv4_1/Relu', 'conv4_2/filter/Initializer/Const', 'conv4_2/filter', 'conv4_2/filter/Assign', 'conv4_2/filter/read', 'conv4_2/L2Loss', 'conv4_2/weight_loss/y', 'conv4_2/weight_loss', 'conv4_2/Conv2D', 'conv4_2/biases/Initializer/Const', 'conv4_2/biases', 'conv4_2/biases/Assign', 'conv4_2/biases/read', 'conv4_2/BiasAdd', 'conv4_2/Relu', 'conv4_3/filter/Initializer/Const', 'conv4_3/filter', 'conv4_3/filter/Assign', 'conv4_3/filter/read', 'conv4_3/L2Loss', 'conv4_3/weight_loss/y', 'conv4_3/weight_loss', 'conv4_3/Conv2D', 'conv4_3/biases/Initializer/Const', 'conv4_3/biases', 'conv4_3/biases/Assign', 'conv4_3/biases/read', 'conv4_3/BiasAdd', 'conv4_3/Relu', 'pool4', 'layer4_out', 'conv5_1/filter/Initializer/Const', 'conv5_1/filter', 'conv5_1/filter/Assign', 'conv5_1/filter/read', 'conv5_1/L2Loss', 'conv5_1/weight_loss/y', 'conv5_1/weight_loss', 'conv5_1/Conv2D', 'conv5_1/biases/Initializer/Const', 'conv5_1/biases', 'conv5_1/biases/Assign', 'conv5_1/biases/read', 'conv5_1/BiasAdd', 'conv5_1/Relu', 'conv5_2/filter/Initializer/Const', 'conv5_2/filter', 'conv5_2/filter/Assign', 'conv5_2/filter/read', 'conv5_2/L2Loss', 'conv5_2/weight_loss/y', 'conv5_2/weight_loss', 'conv5_2/Conv2D', 'conv5_2/biases/Initializer/Const', 'conv5_2/biases', 'conv5_2/biases/Assign', 'conv5_2/biases/read', 'conv5_2/BiasAdd', 'conv5_2/Relu', 'conv5_3/filter/Initializer/Const', 'conv5_3/filter', 'conv5_3/filter/Assign', 'conv5_3/filter/read', 'conv5_3/L2Loss', 'conv5_3/weight_loss/y', 'conv5_3/weight_loss', 'conv5_3/Conv2D', 'conv5_3/biases/Initializer/Const', 'conv5_3/biases', 'conv5_3/biases/Assign', 'conv5_3/biases/read', 'conv5_3/BiasAdd', 'conv5_3/Relu', 'pool5', 'fc6/weights/Initializer/Const', 'fc6/weights', 'fc6/weights/Assign', 'fc6/weights/read', 'fc6/L2Loss', 'fc6/weight_loss/y', 'fc6/weight_loss', 'fc6/Conv2D', 'fc6/biases/Initializer/Const', 'fc6/biases', 'fc6/biases/Assign', 'fc6/biases/read', 'fc6/BiasAdd', 'fc6/Relu', 'dropout/Shape', 'dropout/random_uniform/min', 'dropout/random_uniform/max', 'dropout/random_uniform/RandomUniform', 'dropout/random_uniform/sub', 'dropout/random_uniform/mul', 'dropout/random_uniform', 'dropout/add', 'dropout/Floor', 'dropout/div', 'dropout/mul', 'fc7/weights/Initializer/Const', 'fc7/weights', 'fc7/weights/Assign', 'fc7/weights/read', 'fc7/L2Loss', 'fc7/weight_loss/y', 'fc7/weight_loss', 'fc7/Conv2D', 'fc7/biases/Initializer/Const', 'fc7/biases', 'fc7/biases/Assign', 'fc7/biases/read', 'fc7/BiasAdd', 'fc7/Relu', 'dropout_1/Shape', 'dropout_1/random_uniform/min', 'dropout_1/random_uniform/max', 'dropout_1/random_uniform/RandomUniform', 'dropout_1/random_uniform/sub', 'dropout_1/random_uniform/mul', 'dropout_1/random_uniform', 'dropout_1/add', 'dropout_1/Floor', 'dropout_1/div', 'dropout_1/mul', 'layer7_out', 'init', 'save/Const', 'save/StringJoin/inputs_1', 'save/StringJoin', 'save/num_shards', 'save/ShardedFilename/shard', 'save/ShardedFilename', 'save/SaveV2/tensor_names', 'save/SaveV2/shape_and_slices', 'save/SaveV2', 'save/control_dependency', 'save/MergeV2Checkpoints/checkpoint_prefixes', 'save/MergeV2Checkpoints', 'save/Identity', 'save/RestoreV2/tensor_names', 'save/RestoreV2/shape_and_slices', 'save/RestoreV2', 'save/Assign', 'save/RestoreV2_1/tensor_names', 'save/RestoreV2_1/shape_and_slices', 'save/RestoreV2_1', 'save/Assign_1', 'save/RestoreV2_2/tensor_names', 'save/RestoreV2_2/shape_and_slices', 'save/RestoreV2_2', 'save/Assign_2', 'save/RestoreV2_3/tensor_names', 'save/RestoreV2_3/shape_and_slices', 'save/RestoreV2_3', 'save/Assign_3', 'save/RestoreV2_4/tensor_names', 'save/RestoreV2_4/shape_and_slices', 'save/RestoreV2_4', 'save/Assign_4', 'save/RestoreV2_5/tensor_names', 'save/RestoreV2_5/shape_and_slices', 'save/RestoreV2_5', 'save/Assign_5', 'save/RestoreV2_6/tensor_names', 'save/RestoreV2_6/shape_and_slices', 'save/RestoreV2_6', 'save/Assign_6', 'save/RestoreV2_7/tensor_names', 'save/RestoreV2_7/shape_and_slices', 'save/RestoreV2_7', 'save/Assign_7', 'save/RestoreV2_8/tensor_names', 'save/RestoreV2_8/shape_and_slices', 'save/RestoreV2_8', 'save/Assign_8', 'save/RestoreV2_9/tensor_names', 'save/RestoreV2_9/shape_and_slices', 'save/RestoreV2_9', 'save/Assign_9', 'save/RestoreV2_10/tensor_names', 'save/RestoreV2_10/shape_and_slices', 'save/RestoreV2_10', 'save/Assign_10', 'save/RestoreV2_11/tensor_names', 'save/RestoreV2_11/shape_and_slices', 'save/RestoreV2_11', 'save/Assign_11', 'save/RestoreV2_12/tensor_names', 'save/RestoreV2_12/shape_and_slices', 'save/RestoreV2_12', 'save/Assign_12', 'save/RestoreV2_13/tensor_names', 'save/RestoreV2_13/shape_and_slices', 'save/RestoreV2_13', 'save/Assign_13', 'save/RestoreV2_14/tensor_names', 'save/RestoreV2_14/shape_and_slices', 'save/RestoreV2_14', 'save/Assign_14', 'save/RestoreV2_15/tensor_names', 'save/RestoreV2_15/shape_and_slices', 'save/RestoreV2_15', 'save/Assign_15', 'save/RestoreV2_16/tensor_names', 'save/RestoreV2_16/shape_and_slices', 'save/RestoreV2_16', 'save/Assign_16', 'save/RestoreV2_17/tensor_names', 'save/RestoreV2_17/shape_and_slices', 'save/RestoreV2_17', 'save/Assign_17', 'save/RestoreV2_18/tensor_names', 'save/RestoreV2_18/shape_and_slices', 'save/RestoreV2_18', 'save/Assign_18', 'save/RestoreV2_19/tensor_names', 'save/RestoreV2_19/shape_and_slices', 'save/RestoreV2_19', 'save/Assign_19', 'save/RestoreV2_20/tensor_names', 'save/RestoreV2_20/shape_and_slices', 'save/RestoreV2_20', 'save/Assign_20', 'save/RestoreV2_21/tensor_names', 'save/RestoreV2_21/shape_and_slices', 'save/RestoreV2_21', 'save/Assign_21', 'save/RestoreV2_22/tensor_names', 'save/RestoreV2_22/shape_and_slices', 'save/RestoreV2_22', 'save/Assign_22', 'save/RestoreV2_23/tensor_names', 'save/RestoreV2_23/shape_and_slices', 'save/RestoreV2_23', 'save/Assign_23', 'save/RestoreV2_24/tensor_names', 'save/RestoreV2_24/shape_and_slices', 'save/RestoreV2_24', 'save/Assign_24', 'save/RestoreV2_25/tensor_names', 'save/RestoreV2_25/shape_and_slices', 'save/RestoreV2_25', 'save/Assign_25', 'save/RestoreV2_26/tensor_names', 'save/RestoreV2_26/shape_and_slices', 'save/RestoreV2_26', 'save/Assign_26', 'save/RestoreV2_27/tensor_names', 'save/RestoreV2_27/shape_and_slices', 'save/RestoreV2_27', 'save/Assign_27', 'save/RestoreV2_28/tensor_names', 'save/RestoreV2_28/shape_and_slices', 'save/RestoreV2_28', 'save/Assign_28', 'save/RestoreV2_29/tensor_names', 'save/RestoreV2_29/shape_and_slices', 'save/RestoreV2_29', 'save/Assign_29', 'save/restore_shard', 'save/restore_all']

##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
