



import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler





class singlecell_GLM:
    def __init__(
            self,
            learning_rate=0.001,
            epochs=5000,
            l1_penalty=0.,
            l2_penalty=0.,
        ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.weights = None
        self.loss_history = None
        self.rmse = None

        # for calculating z score, hold onto the mean and std so it can be applied to novel data
        self.X_means = None
        self.X_stds = None


    def _mse(self, y, y_hat):
        """ Mean squared error.
        """
        return np.mean((y - y_hat)**2)


    def _sigmoid(self, z):
        """ Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))
    

    def _softplus(self, z):
        """Softplus activation function.
        """
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z,0)
    

    def _tanh(self, z):
        """ Tanh activation function.
        """
        return np.tanh(z)
    

    def _zscore(self, X):
        """ Z-score for scaling input.
        """

        X_z = np.zeros_like(X)
        savemeans = np.zeros(np.size(X,1))
        savestd = np.zeros(np.size(X,1))

        for feat in range(np.size(X,1)):
            mean_ = np.nanmean(X)
            std_ = np.nanstd(X)
            X_z[:, feat] = (X[:, feat] - mean_) / std_

        return X_z, savemeans, savestd
    

    def _apply_zscore(self, X):

        assert self.X_means is not None, 'Z score has not been computed, so it cannot yet be applied to novel arrays.'
        assert self.X_stds is not None, 'Z score has not been computed, so it cannot yet be applied to novel arrays.'

        X_z = np.zeros_like(X)

        for feat in range(np.size(X,1)):
            X_z[:, feat] = (X[:, feat] - self.X_means[feat]) / self.X_stds[feat]

        return X_z


    def _neg_log_likelihood_loss(self, y, y_hat):
        """ Loss function.

        Loss is calculated as negative log likleihood, with L1 and L2
        regularization terms. If the penalty parameters of the class are
        set to 0, no penalty is applied.
        """

        # Negative log-likelihood
        # 1e-8 is added for numerical stability to avoid log(0)
        nll = np.nanmean(y_hat - y * np.log(y_hat + 1e-8))
        # L1 and L2 penalty terms
        l1 = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
        l2 = self.l2_penalty * np.sum(self.weights[1:] ** 2)

        return nll + l1 + l2
    
    
    def _fit_single(self, X, y, init=0.5, verbose=False):

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        if len(np.shape(X)) != 2:
            X = X[:,np.newaxis]

        weights = np.ones([
            np.size(X,0)+1,
            np.size(y,0)
        ]) * init

        # Add bias term (in a sense, the baseline firing rate?)
        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        m = np.size(y,1)

        loss_history = np.zeros(self.epochs) * np.nan

        for epoch in range(self.epochs):

            y_hat = np.dot(X_bias, weights).T

            # calc loss
            lossval = self._neg_log_likelihood_loss(y, y_hat)
            loss_history[epoch] = lossval

            gradient = np.dot(X_bias.T, (y_hat - y).T) / m
            
            # apply L2 regularization (ridge)
            gradient[1:] += self.l2_penalty * 2 * weights[1:]
            
            # apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(weights)[1:]

            # update weights
            weights -= self.learning_rate * gradient

            mse = self._mse(y, y_hat)

            if verbose and (epoch == 0):
                print('Initial pass:  loss={:.3}  MSE={:.3}'.format(lossval, mse))
            elif verbose and (epoch % 100 == 0):
                print('\rEpoch {}:  loss={:.3}  MSE={:.3}'.format(
                    epoch, lossval, mse), end='', flush=True)
        print('\n')

        return weights, loss_history, mse


    def fit(self, spikes, behavior):
        # behavior should be a 2D array of shape (nVars, nFrames)
        # spikes should be the spike rate for a single cell, with the shape (nFrames)

        n_behaviors = np.size(behavior, 0)
        n_cells = np.size(spikes, 0)

        # 0th index of weights will the the bias term. one set of weights for each cell and each behavior
        all_weights = np.zeros([n_cells, n_behaviors]) * np.nan
        all_loss_histories = np.zeros([n_cells, self.epochs]) * np.nan
        all_rmse = np.zeros(n_cells) * np.nan

        # try to predict all behavior variables from a single spike rate
        y = behavior

        for c in range(n_cells):
           
           X = spikes[c,:].copy() # spike rate

           w, lh, mse = self._fit_single(X, y)
           
           all_weights[c,:] = w
           all_loss_histories[c,:] = lh
           all_rmse[c] = np.sqrt(mse)

        self.weights = all_weights
        self.loss_history = all_loss_histories
        self.rmse = all_rmse


    def _predict(self, X):

        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        # remove this link function, which makes it behave more like a NN than a GLM. Instead,
        # return the identity.
        # y_hat = self._tanh(np.dot(X_bias, self.weights))
        y_hat = np.dot(X_bias, self.weights)

        return y_hat
    

    def make_split(self, X, y, nanfilt=True, test_size=0.25):
        # NaN at any position in X or y will cause issues. Mask out NaNs, which
        # cannot appear in spike data (X), but do show up in the behavior data (y)
        # because of gaps in tracking.
        if nanfilt:
            mask1d = np.sum(np.isnan(y), 0) == 0
            y = y[:,mask1d]
            X = X[:,mask1d]

        train_inds = []
        test_inds = []
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=20)
        for i, (train_index, test_index) in enumerate(ss.split(X.T)):
            train_inds = np.array(sorted(train_index))
            test_inds = np.array(sorted(test_index))

        self.train_inds = train_inds
        self.test_inds = test_inds
        self.nan_mask = mask1d

        X_train = X[:,train_inds]
        y_train = y[:,train_inds]

        X_test = X[:,test_inds]
        y_test = y[:,test_inds]

        return X_train, y_train, X_test, y_test
    

    def get_train_test_inds(self):
        return self.train_inds, self.test_inds, self.nan_mask


    def apply_transform(self, A):

        n_feats = np.size(A,0)

        assert n_feats == len(self.scalers)

        A_scale = np.zeros_like(A) * np.nan

        for feat in range(n_feats):
            A_scale[feat,:] = self.scalers[feat].transform(A[feat,:][:,np.newaxis]).flatten()

        return A_scale


    def fit_apply_transform(self, A):
        # a must be array-like. can be 1D or 2D. could be x or y, but
        # cannot apply to both.

        # for each feature along axis=0, train an independent scaler
        n_feats = np.size(A,0)

        self.scalers = []

        for feat in range(n_feats):
            self.scalers.append(RobustScaler().fit(A[feat,:][:,np.newaxis]))

        A_scale = self.apply_transform(A)

        return A_scale


    def apply_inverse_transform(self, A):

        n_feats = np.size(A,0)

        assert n_feats == len(self.scalers)

        A_invtran = np.zeros_like(A) * np.nan

        for feat in range(n_feats):
            A_invtran[feat,:] = self.scalers[feat].inverse_transform(A[feat,:][:,np.newaxis]).flatten()

        return A_invtran
    

    def score_explained_variance(self, y, y_hat):
        # Similar to r^2 except that this will treat an offset as error, whereas
        # r^2 does not penalize for an offset, it just treats it as an intercept.
        # Could mult by 100 to get a percent. As currently written, max value is 1.0

        # Residual sum of squares
        ss_res = np.sum((y - y_hat)**2)
        # Total variance in y
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / (ss_tot + 1e-8)


    def predict(self, X, y):
        # predict and score weights
        # assume that y is already scaled.

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # should be X_test and y_test as inputs
        y_hat = self._predict(X)
        
        mse = self._mse(y, y_hat.T)
        # explained_variance = self.score_explained_variance(y, y_hat.T)

        return y_hat, mse


    def get_weights(self):
        return self.weights
    

    def get_loss_history(self):
        return self.loss_history
    

    def get_error(self):
        return self.rmse