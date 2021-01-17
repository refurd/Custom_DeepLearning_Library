import numpy as np
from multilayer_nn import activations, losses, optimizers
from pckutils.utils import print_progress


class Mlp:
    
    def __init__(self, optimizer, loss, initializer, regularizer):
        self.optimizer = optimizer
        self.loss = loss
        self.initializer = initializer
        self.regularizer = regularizer 

        self.theta = []
        self.activations = []
        self.gradients = []
        self.sniffer = lambda x: x  # this can be a function for gathering the hidden layers activations

        self.layers = 0 # number of layers
    
    def add_layer(self, nodes, input_length=0, activation=activations.Linear()):
        if input_length==0:
            if len(self.theta) > 0:
                input_length = self.theta[-1].shape[0]
            else:
                assert False, "Missing input_length at first layer!"

        w = self.initializer.create((nodes, input_length))
        self.theta.append(w)
        self.activations.append(activation)
        self.layers += 1
    
    def __forward(self, x):
        '''
        x - an input vector, only one sample
        '''
        w_times_xs = []
        hs = []
        x_current = x
        for w, a in zip(self.theta, self.activations):
            hs.append(x_current)
            w_times_x = np.matmul(w, x_current)
            h = a.activate(w_times_x)
            w_times_xs.append(w_times_x)
            x_current = h
        return x_current, w_times_xs, self.sniffer(hs)

    def __backward(self, yp, w_times_xs, hs, y_real):
        '''
        w_times_xs - w product x for all hidden layers (list)
        hs - outputs of each layer (list)
        y_real - a real output from the training set (one sample)
        '''
        y_predicted = yp
        delta = self.loss.delta_last(y_predicted, y_real) 
        regression_term = self.regularizer.delta_last(self.theta)
        for l in range(self.layers-1, -1, -1): # walking the list backward direction
            wx = w_times_xs[l]
            h = hs[l]
            df = self.activations[l].d_activate
            df_wx = np.matmul(delta, df(wx))
            self.gradients[l] += np.outer(df_wx, h) + regression_term[l]  # calculate gradient for delta
            delta = np.matmul(df_wx, self.theta[l]) # calculate new delta

    def __init_gradients(self):
        if len(self.gradients) == 0:
            for w in self.theta:
                self.gradients.append(np.zeros_like(w))
        else:
            for idx in range(self.layers):
                self.gradients[idx] *= 0.0
    
    def __normalize_gradient_list(self):
        self.gradients = list(map(lambda x: x / np.linalg.norm(x), self.gradients))

    def fit(self, xs, ys, episode, batch_size, verbose=False, callback=None):

        indices = np.array([t for t in range(len(xs))])
        
        for ep in range(episode):

            np.random.shuffle(indices)
            
            ep_length = int(len(xs)/batch_size)
            for itr in range(ep_length):

                # creating a new batch
                batch_x, batch_y = [], []
                for idx in range(itr * batch_size, (itr + 1) * batch_size):
                    idx_ = indices[idx]
                    batch_x.append(xs[idx_])
                    batch_y.append(ys[idx_])
                
                # training step on the batch
                self.__init_gradients()
                batch_y_p = [] # predicted answers for batch_x
                for bx, by in zip(batch_x, batch_y):

                    yp, w_times_xs, hs = self.__forward(bx)
                    self.__backward(yp, w_times_xs, hs, by)
                    batch_y_p.append(yp)
                
                self.__normalize_gradient_list()
                self.optimizer.optimizer_step(self.theta, self.gradients)
                loss = self.loss.loss(batch_y_p, batch_y) + self.regularizer.reg_term(self.theta)
                
                if callback is not None:
                    callback(batch_y_p, batch_y, loss, ep, itr)

                print_progress(ep * ep_length + itr, ep_length * episode, verbose)
    
    def predict(self, x, mode):
        '''
        x - input as a numpy array
        mode - 3 modes are possible:
             onehot > the y vector is given back as a one-hot encoded vector (useful in case of classification)
             index > gives the index of the maximum element in y (in case of classification)
             raw > gives y as it comes out from the network, no further transformation is applied
        '''
        y, _, _ = self.__forward(x)
        if mode == 'raw':
            return y
        elif mode == 'index':
            return np.argmax(y)
        elif mode == 'onehot':
            idx = np.argmax(y)
            y_onehot = np.zeros_like(y)
            y_onehot[idx] = 1
            return y_onehot
        else:
            assert False, "Unknown mode!"
    
    def predict_batch(self, x, mode):
        y = []
        for x_ in x:
            y.append(self.predict(x_, mode))
        return y
