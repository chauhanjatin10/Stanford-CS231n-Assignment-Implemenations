import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X,W)
  for i in range(num_train):
    current_scores = scores[i, :]
    shifted_scores = current_scores - np.max(current_scores)

    loss_i = -shifted_scores[y[i]] + np.log(np.sum(np.exp(shifted_scores)))
    loss += loss_i

    for j in range(num_classes):
      softmax_loss = np.exp(shifted_scores[j])/np.sum(np.exp(shifted_scores))

      if j == y[i]:
        dW[:,j] -= (-1 + softmax_loss)*X[i]
      else:
      dW[:,j] -= softmax_loss*X[i] 


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dw += 2*reg*W 

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X,W)
  shifted_scores = scores - np.max(scores,axis=1)[:,np.newaxis]

  softmax_losses = np.exp(shifted_scores)/np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]

  loss_class = softmax_losses
  loss_class[range(num_train),y] -= 1

  dW = np.dot(X.T, -loss_class)
  dW /= num_train
  dW += 2*reg*W

  correct_class_scores = np.choose(y , shifted_scores.T)
  loss = -correct_class_scores + np.log(np.sum(np.xp(shifted_scores), axis=1))
  loss = np.sum(loss)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

