from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        '''Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.

        '''
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

        out_L1, cache_L1 = affine_relu_forward(X, W1, b1)
        out_L2, cache_L2 = affine_forward(out_L1, W2, b2)
        scores = out_L2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        weight_sum = np.sum(np.square(W1)) + np.sum(np.square(W2))
        regularization = 0.5 * self.reg * weight_sum

        loss, dx = softmax_loss(scores, y)
        loss += regularization

        dx2, dw2, db2 = affine_backward(dx, cache_L2)
        dx1, dw1, db1 = affine_relu_backward(dx2, cache_L1)

        # When using L2 regularization, every weight is decayed linearly towards
        # zero during backpropagation
        dw2 += self.reg * W2
        dw1 += self.reg * W1

        # Add the new key-value pairs to the `grads` dictionary
        grads.update({'W1': dw1,
                      'b1': db1,
                      'W2': dw2,
                      'b2': db2})

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

        '''Questions:
        1. Shouldn't the ReLU activation function be used in the final layer (prior
           to evaluating the softmax loss)?
        2. How do we update the gradients with respect to the parameters W1 and W2
           with respect to L2 regularization?

        '''


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        all_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, len(all_dims)):
            prev_layer_dims = all_dims[i - 1]
            next_layer_dims = all_dims[i]
            weight_name, bias_name = 'W{}'.format(i), 'b{}'.format(i)

            # Store params
            self.params[weight_name] = np.random.normal(0.0, weight_scale, (prev_layer_dims, next_layer_dims))
            self.params[bias_name] = np.zeros(next_layer_dims)

            # If using batch normalization, store the scale and shift params -
            # remember that the outputs of the last layer should not be normalized
            if self.use_batchnorm and i < self.num_layers:
                gamma_name, beta_name = 'gamma{}'.format(i), 'beta{}'.format(i)

                self.params[gamma_name] = np.ones(next_layer_dims)
                self.params[beta_name] = np.ones(next_layer_dims)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        input = X
        caches = []
        for i in range(1, self.num_layers): # 1, 2, 3, 4
            weight_name, bias_name = 'W{}'.format(i), 'b{}'.format(i)
            w = self.params[weight_name]
            b = self.params[bias_name]

            out, cache = affine_relu_forward(input, w, b)

            # Store all of the cached variables for the backward pass
            caches.append(cache)

            # The input to the next layer is the output of this layer
            input = out

        # Compute the final output layer
        w_final = self.params['W{}'.format(self.num_layers)]
        b_final = self.params['b{}'.format(self.num_layers)]
        out_final, cache_final = affine_forward(input, w_final, b_final)
        caches.append(cache_final)

        scores = out_final

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Compute softmax loss
        loss, dx = softmax_loss(scores, y)

        weight_sum = np.sum(np.square(w_final))

        # Compute the gradients with respect to the final output layer
        dx_out, dw_out, db_out = affine_backward(dx, caches[-1])
        dw_out += self.reg * w_final

        grads.update({'W{}'.format(self.num_layers): dw_out,
                      'b{}'.format(self.num_layers): db_out})

        # Backpropagation - keep track of the sum of all weights in the
        # network
        output = dx_out
        for i in reversed(range(1, self.num_layers)): # 4, 3, 2, 1
            weight_name, bias_name = 'W{}'.format(i), 'b{}'.format(i)
            w = self.params[weight_name]
            b = self.params[bias_name]

            # Accumulate regularization loss
            weight_sum += np.sum(np.square(w))

            dx, dw, db = affine_relu_backward(output, caches[i - 1])

            # When using L2 regularization, every weight is decayed linearly towards
            # zero during backpropagation
            dw += self.reg * w

            # Add gradients to dictionary
            grads.update({weight_name: dw,
                          bias_name: db})

            output = dx

        # Calculate the total regularization loss
        regularization = 0.5 * self.reg * weight_sum
        loss += regularization

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
