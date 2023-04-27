## Convolutional Neural Networks: Step by Step
"""
Welcome to Course 4's first assignment! In this assignment, you will implement convolutional (CONV) and pooling (POOL)
layers in numpy, including both forward propagation and (optionally) backward propagation.

By the end of this notebook, you'll be able to:

Explain the convolution operation
Apply two different types of pooling operation
Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
Build a convolutional neural network
"""

#1 - Packages
"""
Let's first import all the packages that you will need during this assignment.

numpy is the fundamental package for scientific computing with Python.
matplotlib is a library to plot graphs in Python.
np.random.seed(1) is used to keep all the random function calls consistent. This helps to grade your work.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)


#2 - Outline of the Assignment
"""
You will be implementing the building blocks of a convolutional neural network! Each function you will implement will
have detailed instructions to walk you through the steps:

. Convolution functions, including:
    . Zero Padding
    . Convolve window
    . Convolution forward
    . Convolution backward (optional)
. Pooling functions, including:
    . Pooling forward
    . Create mask
    . Distribute value
    . Pooling backward (optional)
This notebook will ask you to implement these functions from scratch in numpy. In the next notebook, you will use the
TensorFlow equivalents of these functions to build the model.
"""

"""
Note: For every forward function, there is a corresponding backward equivalent. Hence, at every step of your forward
module you will store some parameters in a cache. These parameters are used to compute gradients during backpropagation.
"""

#3 - Convolutional Neural Networks
"""
Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand
in Deep Learning. A convolution layer transforms an input volume into an output volume of different size.

In this part, you will build every step of the convolution layer. You will first implement two helper functions: one
for zero padding and the other for computing the convolution function itself.
"""

#3.1 - Zero-Padding
"""
Zero-padding adds zeros around the border of an image:

The main benefits of padding are:

It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important
for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important
special case is the "same" convolution, in which the height/width is exactly preserved after one layer.

It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer
would be affected by pixels at the edges of an image.
"""

#Exercise 1 - zero_pad
"""
Implement the following function, which pads all the images of a batch of examples X with zeros. Use np.pad. Note if
you want to pad the array "a" of shape  (5,5,5,5,5)
with pad = 1 for the 2nd dimension, pad = 3 for the 4th dimension and pad = 0 for the rest, you would do:

a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), mode='constant', constant_values = (0,0))
"""

# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    #(≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
    # YOUR CODE ENDS HERE
    return X_pad

#<Test>
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)
plt.show()
#<Test/>



#3.2 - Single Step of Convolution
"""
In this part, implement a single step of convolution, in which you apply the filter to a single position of the input.
This will be used to build a convolutional unit, which:

. Takes an input volume
. Applies a filter at every position of the input
. Outputs another volume (usually of different size)


In a computer vision application, each value in the matrix on the left corresponds to a single pixel value. You
convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing
them up and adding a bias. In this first step of the exercise, you will implement a single step of convolution,
corresponding to applying a filter to just one of the positions to get a single real-valued output.

Later in this notebook, you'll apply this function to multiple positions of the input to implement the full
convolutional operation.
"""

#Exercise 2 - conv_single_step
"""
Implement conv_single_step().
"""
# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    # YOUR CODE STARTS HERE
    s = a_slice_prev*W
    Z = np.float64(sum(sum(sum(s))))
    Z+=np.float64(b)
    # YOUR CODE ENDS HERE
    return Z

#<Test>
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"
#<Test/>


#3.3 - Convolutional Neural Networks - Forward Pass
"""
In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D
matrix output. You will then stack these outputs to get a 3D volume.
"""

#Exercise 3 - conv_forward
"""
Implement the function below to convolve the filters W on an input activation A_prev.
This function takes the following inputs:

A_prev, the activations output by the previous layer (for a batch of m inputs);
Weights are denoted by W. The filter window size is f by f.
The bias vector is b, where each filter has its own (single) bias.
You also have access to the hyperparameters dictionary, which contains the stride and the padding.


Hint:

To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:

a_slice_prev = a_prev[0:2,0:2,:]
Notice how this gives a 3D slice that has height 2, width 2, and depth 3. Depth is the number of channels.
This will be useful when you will define a_slice_prev below, using the start/end indexes you will define.

To define a_slice you will need to first define its corners vert_start, vert_end, horiz_start and horiz_end.
"""

# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape (≈1 line)
    # (m, n_H_prev, n_W_prev, n_C_prev) = None
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape (≈1 line)
    # (f, f, n_C_prev, n_C) = None
    (f, f, n_C_prev, n_C) = W.shape
    # Retrieve information from "hparameters" (≈2 lines)
    # stride = None
    # pad = None
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # n_H = None
    # n_W = None
    n_H = int((n_H_prev-f+2*pad)/stride)+1
    n_W = int((n_W_prev-f+2*pad)/stride)+1
    # Initialize the output volume Z with zeros. (≈1 line)
    # Z = None
    Z=np.zeros([m,n_H,n_W,n_C])
    # Create A_prev_pad by padding A_prev
    # A_prev_pad = None
    A_prev_pad = zero_pad(A_prev, pad)
    # for i in range(None):               # loop over the batch of training examples
        # a_prev_pad = None               # Select ith training example's padded activation
        # for h in range(None):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            # for w in range(None):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                # for c in range(None):   # loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    # a_slice_prev = None
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    # weights = None
                    # biases = None
                    # Z[i, h, w, c] = None
    # YOUR CODE STARTS HERE
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]               # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            vert_start = h+h*(stride-1)
            vert_end = h+h*(stride-1)+f
            for w in range(n_W):       # loop over horizontal axis of the output volume
                horiz_start = w+w*(stride-1)
                horiz_end = w+w*(stride-1)+f
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)#(sum(sum(sum(weights*a_slice_prev))).reshape(biases.shape)+biases).sum()
    # YOUR CODE ENDS HERE
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache

#<Test>
np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
z_mean = np.mean(Z)
z_0_2_1 = Z[0, 2, 1]
cache_0_1_2_3 = cache_conv[0][1][2][3]
print("Z's mean =\n", z_mean)
print("Z[0,2,1] =\n", z_0_2_1)
print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)
conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
conv_forward_test_2(conv_forward)
#<Test/>

"""
Finally, a CONV layer should also contain an activation, in which case you would add the following line of code:

# Convolve the window to get back one output neuron
Z[i, h, w, c] = ...
# Apply activation
A[i, h, w, c] = activation(Z[i, h, w, c])
You don't need to do it here, however.
"""

#4 - Pooling Layer
"""
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make
feature detectors more invariant to its position in the input. The two types of pooling layers are:

Max-pooling layer: slides an ( 𝑓,𝑓) window over the input and stores the max value of the window in the output.

Average-pooling layer: slides an ( 𝑓,𝑓) window over the input and stores the average value of the window in the output.
"""

"""
These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the
window size  𝑓. This specifies the height and width of the  𝑓×𝑓 window you would compute a max or average over.
"""

#4.1 - Forward Pooling
"""
Now, you are going to implement MAX-POOL and AVG-POOL, in the same function.
"""

#Exercise 4 - pool_forward
"""
Implement the forward pass of the pooling layer. Follow the hints in the comments below.
"""

# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    # Initialize output matrix A
    A = np.zeros([m, n_H, n_W, n_C])
    # for i in range(None):                         # loop over the training examples
        # for h in range(None):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            # for w in range(None):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                # for c in range (None):            # loop over the channels of the output volume
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    # a_prev_slice = None
                    # Compute the pooling operation on the slice.
                    # Use an if statement to differentiate the modes.
                    # Use np.max and np.mean.
                    # if mode == "max":
                        # A[i, h, w, c] = None
                    # elif mode == "average":
                        # A[i, h, w, c] = None
    # YOUR CODE STARTS HERE
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            vert_start = h+h*(stride-1)
            vert_end = h+h*(stride-1)+f
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                horiz_start = w+w*(stride-1)
                horiz_end = w+w*(stride-1)+f
                for c in range (n_C_prev):            # loop over the channels of the output volume
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    # YOUR CODE ENDS HERE
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

#<Test>
# Case 1: stride of 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])

pool_forward_test(pool_forward)



# Case 2: stride of 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[0] =\n", A[0])
print()

A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1] =\n", A[1])
#<Test/>

"""
What you should remember:

. A convolution extracts features from an input image by taking the dot product between the input data and a 3D array
  of weights (the filter).

. The 2D output of the convolution is called the feature map

. A convolution layer is where the filter slides over the image and computes the dot product
  This transforms the input volume into an output volume of different size

. Zero padding helps keep more information at the image borders, and is helpful for building deeper networks, because
  you can build a CONV layer without shrinking the height and width of the volumes

. Pooling layers gradually reduce the height and width of the input by sliding a 2D window over each specified region,
  then summarizing the features in that region

"""
