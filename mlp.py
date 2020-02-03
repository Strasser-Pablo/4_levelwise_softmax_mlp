"""
MLP with levelwise derivatives.

Author: Joao A. Candido Ramos

/!\IMPORTANT/!\:
This code is based on Assignment 1 given during the course "CS231n: 
Convolutional Neural Networks for Visual Recognition" at Stanford University.
"""
import numpy as np


class MLP:
    def __init__(self, dims, hidden_size, num_classes, std=1e-3, random_seed=0):
        """

        """
        self.num_classes = num_classes
        if random_seed:
            np.random.seed(random_seed)
        self.W1 = std * np.random.randn(dims, hidden_size)
        self.W2 = std * np.random.randn(hidden_size, num_classes)

    def forward(self, X):
        """
        Computes the forward pass, takes data as input and output probabilities.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.

        Returns:
        - The probabilities of each sample to become to a given class.
        """
        #######################################################################
        # TODO: Compute the scores and apply the softmax, scores should be
        # stored in self.scores and the result of the softmax in 
        # self.softmax_output
        #######################################################################
        self.hidden = None
        self.scores = None

        # softmax
        self.probs = None
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return self.probs

    def loss(self, X, y, reg=0.0):
        """
        Computes the loss.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - reg: (float) regularization strength.

        Returns:
        - the loss
        """
        loss = None
        # get the number of samples
        N = X.shape[0]
        #######################################################################
        # TODO: Compute the loss. You should get the probabilities calling 
        # self.forward and not implementing it there again
        #######################################################################
        pass      
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return loss

    def backward(self, X, y, reg, dP_dS=False):
        """
        Computes the gradients for each level of the graph.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - reg: (float) regularization strength.

        Returns:
        - DLoss / dWeights
        """
        grads = {}
        scores = None
        N = X.shape[0]
        #######################################################################
        # TODO: Find the local derivatives and the gradients at each level.
        # Replace the np.zeros(1) values by your derivations.
        #######################################################################
        # Y one hot
        y_onehot = None

        # LEVEL 1
        # dLoss / dProbs
        self.dLoss_dProbs = np.zeros(1)

        # LEVEL 2
        # dProbs / dScores
        # IMPORTANT: This is not trivial to do in a vectorized manner you are
        # allowed to use as many loops as you want inside the if
        # IF YOU IMPLEMENT THIS DERIVATIVE WITHOUT LOOPS YOU HAVE A BONUS !!! =)
        #######################################################################
        # LOOP ZONE
        self.dProbs_dScores = np.zeros(1)
        if dP_dS:
            pass
        # END OF LOOP ZONE
        #######################################################################

        # dLoss / dScores
        # Since it's not easy to vectorize dProbs / dScores, do not use the 
        # chain rule there !!! Prefer the analytical solution !
        self.dLoss_dScores = np.zeros(1)

        # LEVEL 3
        # dScores / dW2
        self.dScores_dW2 = np.zeros(1)

        # dLoss / dW2 (by chain rule)
        self.dLoss_dW2 = np.zeros(1)
        
        # dScores / dHidden
        self.dScores_dHidden = np.zeros(1)

        # dLoss / dHidden (by chain rule)
        self.dLoss_dHidden = np.zeros(1)

        # LEVEL 4
        # dHidden / dW1
        self.dHidden_dW1 = np.zeros(1)

        # dLoss / dW1 (by chain rule)
        self.dLoss_dW1 = np.zeros(1)

        # dHidden / dX
        self.dHidden_dX = np.zeros(1)

        # dLoss / dHidden (by chain rule)
        self.dLoss_dX = np.zeros(1)

        # REGULARIZER
        # add regularization to the gradients of the weights 2

        # add regularization to the gradients of the weights 1

        # Fill grads
        grads["W1"] = None
        grads["W2"] = None
        # #######################################################################
        # #--------------------------- END OF YOUR CODE -------------------------
        # #######################################################################

        return grads

    def train(self, X, y, num_iters=100, learning_rate=0.0,
              reg=0.0, batch_size=10, verbose=True):
        """
        Train the softmax classifier using stochastic gradient descent (SGD).

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - X_val: A numpy array of shape (M, D) containing validation data (cf. X)
        - y_val: A numpy array of shape (M,) containing validation labels (cf. y)
        - num_iters: (integer) number of steps to take when optimizing
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Returns:
        - A dictionary with the history of losses and accuracies for training 
        and validation data
        """
        history = {}
        history["train_loss"] = []
        history["train_acc"] = []
        
        N, C = X.shape
        max_iterations = N // batch_size

        for it in range(1, num_iters+1):
            batch_indices = np.random.choice(N, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            probs = self.forward(X_batch)
            loss = self.loss(X_batch, y_batch, reg)
            grads = self.backward(X_batch, y_batch, reg)

            ###############################################################
            # TODO: optimize with stochastic gradient descent (SGD)
            ###############################################################
            pass
            ###############################################################
            #----------------------- END OF YOUR CODE ---------------------
            ###############################################################

            # Decay learning rate
            #learning_rate *= learning_rate_decay

            # keep a trace of training
            history["train_loss"].append(loss)
            history["train_acc"].append((self.predict(X) == y).mean())

            if verbose:
                print("[TRAIN] iteration {:03d}/{:03d}, loss: {:.5f}, accuracy: {:.5f}".format(it, num_iters, history["train_loss"][-1], history["train_acc"][-1]), end="\r")
        if verbose:
            print(''.ljust(150), end='\r')
        return history

    def predict(self, X):
        """
        Computes the predictions given the probs, basically find the index of 
        highest prob.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.

        Returns:
        - y_pred: A numpy array of shape (N, 1) containing the classes predicted
        """
        y_pred = None
        #######################################################################
        # TODO: use the forward pass to find the predictions.
        #######################################################################
        pass
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return y_pred
