from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        C, H, W = input_dim

        # Convolutional layer
        self.params['W1'] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        # Hidden affine layer
        stride = 1
        pad = (filter_size - 1) // 2
        conv_output_h = (H + 2 * pad - filter_size) // stride + 1
        conv_output_w = (W + 2 * pad - filter_size) // stride + 1

        pool_dim = 2
        stride_pool = 2
        pool_output_h = (conv_output_h - pool_dim) // stride_pool + 1
        pool_output_w = (conv_output_w - pool_dim) // stride_pool + 1

        pool_output_flattened = num_filters * pool_output_h * pool_output_w

        self.params['W2'] = np.random.normal(0.0, weight_scale, (pool_output_flattened, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        # Output affine layer
        self.params['W3'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        c_out, c_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)  # Convolutional layer
        h_out, h_cache = affine_relu_forward(c_out, W2, b2);                        # Hidden affine layer
        f_out, f_cache = affine_forward(h_out, W3, b3);                             # Final affine layer

        scores = f_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dx = softmax_loss(scores, y)

        regularization_loss = 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        loss += regularization_loss

        dx_f, dw_f, db_f = affine_backward(dx, f_cache)
        dx_h, dw_h, db_h = affine_relu_backward(dx_f, h_cache)
        dx_c, dw_c, db_c = conv_relu_pool_backward(dx_h, c_cache)

        grads.update({'W1': dw_c,
                      'b1': db_c,
                      'W2': dw_h,
                      'b2': db_h,
                      'W3': dw_f,
                      'b3': db_f})

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
