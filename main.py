import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

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
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
 
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
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

    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output0 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output1 = tf.add(conv_1x1, output0)
    output1 = tf.layers.conv2d_transpose(output1, num_classes, 4, 2, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output2 = tf.add(conv_1x1, output1)
    output2 = tf.layers.conv2d_transpose(output2, num_classes, 16, 8, padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output2
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
    # setup logits
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # setup loss
    entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    regularizer_loss = tf.losses.get_regularization_loss()
    total_loss = entropy_loss + regularizer_loss
    # setup train_op
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss)
    return logits, train_op, total_loss
tests.test_optimize(optimize)

MEAN_IOU_INTERVAL=5
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             logits, correct_label, num_classes, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    learning_curve = []
    mean_iou_curve = []
    labels = tf.argmax(tf.reshape(correct_label, (-1, num_classes)), axis=1)
    predictions = tf.argmax(logits, axis=1)
    mean_iou, update_op = tf.metrics.mean_iou(labels=labels, predictions=predictions, num_classes=num_classes)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # TODO: Implement function
    for epoch in range(epochs):
        step = 0
        for image, label in get_batches_fn(batch_size):
            _, total_loss_val = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:image, correct_label:label, keep_prob:0.5, learning_rate:1e-3})
            print("\t[%03d-%03d] loss=%f"%(epoch, step, total_loss_val))
            step += 1

        learning_curve.append(total_loss_val)
        if epoch%MEAN_IOU_INTERVAL == 0:
            mean_iou_vals = [] 
            for image, label in get_batches_fn(batch_size):
                _ = sess.run(update_op, feed_dict={input_image:image, correct_label:label, keep_prob:1.0, learning_rate:1e-3})
                mean_iou_vals.append(sess.run(mean_iou, feed_dict={input_image:image, correct_label:label, keep_prob:1.0, learning_rate:1e-3}))
            mean_iou_val = sum(mean_iou_vals)/len(mean_iou_vals)
            print("[%03d] mean_iou=%f "%(epoch, mean_iou_val))
            mean_iou_curve.append(mean_iou_val)

            if mean_iou_val > 0.83:
                break;

    return learning_curve, mean_iou_curve
#tests.test_train_nn(train_nn)

print("#################################################################################")
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

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, total_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 175
        batch_size = 16 # Tuning Params
        learning_curve, mean_iou_curve = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, total_loss, input_image, logits, correct_label, num_classes, keep_prob, learning_rate)

        plt.plot(learning_curve)
        plt.title('training curves')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.savefig('./learning_curve.png')

        plt.plot([i*MEAN_IOU_INTERVAL for i in range(len(mean_iou_curve))], mean_iou_curve)
        plt.title('mean iou curves')
        plt.ylabel('iou')
        plt.xlabel('epochs')
        plt.savefig('./mean_iou_curve.png')
        #plt.show()

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        output_dir = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        video_file = 'output.mp4'
        fps = 2
        print("Creating video {}, FPS={}".format(video_file, fps))
        clip = ImageSequenceClip(output_dir, fps=fps)
        clip.write_videofile(video_file)


if __name__ == '__main__':
    run()
