import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def show_weights_histogram(model, name='block1_conv1'):
    w = model.get_layer(name).get_weights()[0]
    plt.hist(w.flatten())
    plt.title("w values after 'imagenet' weights are loaded")
    plt.show()


def get_image_data(path, resize=None, dtype=np.float32):
    filepath = glob.glob(path)[0]
    image = cv2.imread(filepath).astype(dtype)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image Shape = ", image.shape)

    ratio = None
    if resize != None:
        if type(resize) == int:
            height, width, n_channel = image.shape

            min_size = width if height > width else height

            ratio = resize / min_size

            resize = (int(width * ratio), int(height * ratio))

        else:
            assert type(resize) == tuple, "If 'resize' is not 'int', then should be 'tuple' = (resize_width, resize_height)"

            resize = (resize[1], resize[0])

        image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)

    return image, ratio


def preprocessing(image):
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939

    return image


def deprocessing(image):
    image[:, :, 0] += 123.68
    image[:, :, 1] += 116.779
    image[:, :, 2] += 103.939

    image = np.clip(image, 0, 255).astype('uint8')

    return image


def Gram_Matrix(input_tensor):
    # input_tensor Shape : (height, width, n_channel)
    # We should reshape to (height * width, n_channel)
    assert len(input_tensor.shape) == 3, "'input_tensor' shape is wrong. 'input_tensor' should be shape : (height, width, n_channel)"

    n_channel = input_tensor.shape[-1]
    input_tensor = tf.reshape(input_tensor, shape=(-1, n_channel))

    # matmul(input.T, input)  ==> Gamma Matrix
    gram_matrix = tf.matmul(tf.transpose(input_tensor), input_tensor)  # Shape : (n_channel, n_channel)

    return gram_matrix


def total_variation_loss(x):
    _, height, width, _ = x.shape

    height = height.value
    width = width.value

    a = tf.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = tf.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])

    return tf.reduce_sum(tf.pow(a + b, 1.25))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image_path', type=str, default='./images/content/content_image.*',
                        help="Path to the content image. ex)'./images/content/content_image.jpg'")

    parser.add_argument('--style_image_path', type=str, default='./images/style/style_image.*',
                        help="Path to the style image. ex) './images/style/style_image.jpg'")

    parser.add_argument('--model_type', type=int, default=0,
                        help="Options are 0 == VGG16, 1 == VGG19.")

    parser.add_argument('--image_resize', type=int, default=512,
                        help="If image_resize == 'int', min(height, width) of images = image_resize.\
                              If image_resize == 'tuple',  (height, width) of images = image_resize.")

    parser.add_argument('--rescale_image', type=bool, default=False,
                        help="Rescale final image to original size.")

    parser.add_argument('--content_blocks', type=str, default=['block4_conv2'],
                        help="Layer list for feature vector of Content image.")

    parser.add_argument('--style_blocks', type=str,
                        default=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
                        help='Layer list for feature vector of Style image.')

    parser.add_argument('--loss_ratio', type=float, default=1e-3,
                        help='alpha / beta for loss function.')

    parser.add_argument("--total_variation_weight", type=float, default=0,
                        help="Total Variation weight. Default : 8.5e-5")

    parser.add_argument('--initial_type', type=str, default='random',
                        help="Options are 'content', 'style', 'random'.")

    parser.add_argument('--optimizer_type', type=int, default=1,
                        help='Options are 0 == Adam Optimizer, 1 == L-BFGS-B Optimizer.')

    parser.add_argument('--learning_rate', type=float, default=1e+1, help='-')

    parser.add_argument('--beta_1', type=float, default=0.9, help='-')

    parser.add_argument('--beta_2', type=float, default=0.999, help='-')

    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')

    parser.add_argument('--iteration', type=int, default=150, help='-')

    args, unknown = parser.parse_known_args()

    """ HYPER PARAMETER  """
    content_img_path    = args.content_image_path
    style_img_path      = args.style_image_path
    image_resize        = args.image_resize
    rescale_image       = args.rescale_image
    content_blocks      = args.content_blocks
    style_blocks        = args.style_blocks
    style_weights       = {block: 1 / len(style_blocks) for block in style_blocks}  # The factor 'w' was always equal to one divided by the number of active layers in the paper
    loss_ratio          = args.loss_ratio  # alpha / beta
    tv_weight           = args.total_variation_weight
    initial_type        = args.initial_type
    model_type          = args.model_type
    optimizer_type      = args.optimizer_type
    learning_rate       = args.learning_rate
    beta_1              = args.beta_1
    beta_2              = args.beta_2
    epsilon             = args.epsilon
    iteration           = args.iteration

    """ Content, Style, Generated Image """
    # generated_image is trainable parameter. Initialize by random_normal noise.
    content_image, rescale = get_image_data(content_img_path, image_resize)
    style_image, _ = get_image_data(style_img_path, content_image.shape[:2])

    print("After Image Shape", content_image.shape)

    if initial_type == 'content':
        generated_image = content_image.copy()
    elif initial_type == 'style':
        generated_image = style_image.copy()
    elif initial_type == 'random':
        generated_image = tf.random_normal(shape=content_image.shape, stddev=np.std(content_image))

    generated_image = tf.Variable(generated_image, dtype=tf.float32, name='random_noise', trainable=True)

    # preprocessing - subtract mean rgb value of 'imagenet'
    content_image = preprocessing(content_image)
    style_image   = preprocessing(style_image)

    # Reshape to 1 batch image and convert to Tensor.
    image_shape = (1,) + content_image.shape  # shape = (1, height, width, 3)

    content_image = content_image.reshape(image_shape)
    style_image   = style_image.reshape(image_shape)
    init_tensor   = tf.reshape(generated_image, shape=image_shape)

    # Load pretrained model using Keras API
    with tf.variable_scope('pretrained_model'):
        if model_type == 0:
            model = VGG16(weights='imagenet', input_tensor=init_tensor, include_top=False)
        elif model_type == 1:
            model = VGG19(weights='imagenet', input_tensor=init_tensor, include_top=False)

        keras_variables = [var.name for var in tf.global_variables() if 'pretrained_model' in var.name]

    # Output Tensor of Keras model into Dictionary
    output_dict = {layer.name: layer.output for layer in model.layers}

    # Session
    sess = K.get_session()
    K.set_session(sess)

    # Get Content feature and Style feature
    Ps = {}
    As = {}

    for block in content_blocks:
        feature_vectors = sess.run(output_dict[block], feed_dict={init_tensor: content_image})[0]
        Ps[block] = tf.constant(feature_vectors, dtype=tf.float32)  # feature vector of Content image

    for block in style_blocks:
        feature_vectors = sess.run(output_dict[block], feed_dict={init_tensor: style_image})[0]
        As[block] = Gram_Matrix(feature_vectors)  # Gram Matrix of Style feature vector

    """ 
    Loss  
    = alpha * content_loss + beta * style_loss (alpha/beta = loss_ratio)

    my code)
    loss = loss_ratio * content_loss + style_loss
    """

    # Content Loss
    content_loss = 0
    for block in content_blocks:
        F = output_dict[block][0]  # feature vector of Generated iamge
        P = Ps[block]  # feature vector of Content image

        content_loss += 1 / 2 * tf.reduce_sum(tf.pow((F - P), 2))

    # Style Loss
    style_loss = 0
    for block in style_blocks:
        F = output_dict[block][0]
        A = As[block]  # Gram Matrix of Style feature vector
        G = Gram_Matrix(F)  # Gram Matrix of Generated feature vector

        height, width, n_channel = F.shape
        size = height.value * width.value
        scale = 1 / (4 * (n_channel.value ** 2) * (size ** 2))
        w = style_weights[block]

        style_loss += w * scale * tf.reduce_sum(tf.pow((G - A), 2))

    # Total Variation Loss
    tv_loss = tv_weight * total_variation_loss(init_tensor)

    loss = loss_ratio * content_loss + style_loss + tv_loss

    # Minimize cost
    trainble_variables = [var for var in tf.global_variables() if 'pretrained_model' not in var.name]  # Should not train the weights of pretrained model.
    if optimizer_type == 0:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2,
                                           epsilon=epsilon).minimize(loss, var_list=trainble_variables)
    elif optimizer_type == 1:
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=trainble_variables, method='L-BFGS-B',
                                                           options={'maxiter': iteration})

    # Initialize
    uninitialize_variables = [var for var in tf.global_variables() if var.name not in keras_variables]
    sess.run(tf.variables_initializer(uninitialize_variables))

    # Make sure that pretrained model's weights are initialized. They should never be initialized.
    # show_weights_histogram(model)

    # Training
    with sess.as_default():

        if optimizer_type == 0:  # Adam Optimizer
            for i in range(iteration):
                _cost, _c_cost, _s_cost, _tv_cost, _ = sess.run([loss, content_loss, style_loss, tv_loss, optimizer])

                if i % ((iteration // 10)) == 0:
                    print('iter : {}'.format(i + 1), 'total loss : {:.2f}'.format(_cost),
                          'content_loss : {:.2f}'.format(_c_cost), 'style_loss : {:.2f}'.format(_s_cost))

        if optimizer_type == 1:  # L-BFGS-B Optimizer
            _iter = 0


            def callback(_cost, _c_cost, _s_cost, _tv_loss):
                global _iter

                if _iter % ((iteration // 10)) == 0:
                    print('iter : {}'.format(_iter + 1), 'total loss : {:.2f}'.format(_cost),
                          'content_loss : {:.2f}'.format(_c_cost), 'style_loss : {:.2f}'.format(_s_cost),
                          'tv_loss : {:.2f}'.format(_tv_loss))

                _iter += 1


            optimizer.minimize(sess, fetches=[loss, content_loss, style_loss, tv_loss], loss_callback=callback)

    print("Complete Sytle Transfer!")

    generated_image = sess.run(init_tensor)[0]

    # deprocessing - add mean rgb value of 'imagenet'
    generated_image = deprocessing(generated_image)

    if rescale_image == True:
        generated_image = cv2.resize(generated_image, None, fx=1 / rescale, fy=1 / rescale,
                                     interpolation=cv2.INTER_CUBIC)

    print("Final Image Shape =", generated_image.shape)

    # Save image.
    save_name = os.path.basename(content_img_path)[:-4].replace('-', '_') + '_' + os.path.basename(style_img_path)[:-4].replace('-', '_')
    print('Final Image name =', save_name)

    cv2.imwrite('./images/sample/{}.jpg'.format(save_name), cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))

    K.clear_session()
