import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

class Whitens(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=np.finfo(np.float64).eps, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, method='zca', y=None):
        """Compute the mean, whitening and dewhitening matrices using specified whitening method.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
        """
        regularization_term = max(self.regularization, 1e-5)  # Adjust 1e-5 based on your needs

        # Validates the input array X (checks if 2D, converts to float, etc.)
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        ensure_2d=True)
        # Convert the array X to a floating-point type array
        X = as_float_array(X, copy=self.copy)
        # Calculate the mean of each feature in X
        self.mean_ = X.mean(axis=0)
        # Subtract the mean from X for centering the data
        X_ = X - self.mean_
        # Compute the covariance matrix of the centered data
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        # Compute the covariance matrix of the centered data
        if method in ['zca', 'pca', 'cholesky']:
            Lambda, U = np.linalg.eigh(cov) #Compute the covariance matrix of the centered data
            Lambda = np.flip(Lambda)         # Reverse the order of eigenvalues and corresponding eigenvectors
            U = np.flip(U,axis=1)
            U = np.sign(np.diag(U))*U          # Adjust the signs of eigenvectors for consistency
            Lambda += regularization_term
            s = np.sqrt(Lambda)
            s_inv = np.diag(1. / s)
            s = np.diag(s)
            # Compute the whitening and dewhitening matrices for each method
            if method == 'zca':
                self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
                self.dewhiten_ = np.dot(np.dot(U, s), U.T)
            elif method =='pca':
                self.whiten_ = np.dot(s_inv, U.T)
                self.dewhiten_ = np.dot(U, s)
            elif method == 'cholesky':
                self.whiten_ = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1./Lambda.clip(self.regularization)), U.T))).T
                self.dewhiten_ = np.dot(cov,self.whiten_.T)

        elif method in ['zca_cor', 'pca_cor']:
            V_sqrt = np.std(X, axis=0) # Compute the standard deviation of each feature in X
            # Create diagonal matrices for the inverse and the standard deviation of each feature
            V_inv = np.diag(1./V_sqrt)
            V_sqrt = np.diag(V_sqrt)
            # Compute the normalized covariance matrix
            P = np.dot(np.dot(V_inv, cov), V_inv)
            # Calculate the eigenvalues and eigenvectors of the normalized covariance matrix
            Theta, G = np.linalg.eigh(P)
            # Reverse the order of eigenvalues and corresponding eigenvectors
            Theta = np.flip(Theta)
            G = np.flip(G,axis=1)
            # Adjust the signs of eigenvectors for consistency
            G = np.sign(np.diag(G))*G
            # Calculate the square root of eigenvalues, adding a small regularization term to avoid division by zero
            Theta += regularization_term
            p = np.sqrt(Theta.clip(self.regularization))
            # Create diagonal matrices for the inverse and the square root of eigenvalues
            p_inv = np.diag(1./p)
            p = np.diag(p)
            if method == 'zca_cor':
                self.whiten_ = np.dot(np.dot(np.dot(G, p_inv), G.T), V_inv)
                self.dewhiten_ = np.dot(V_sqrt,np.dot(np.dot(G, p), G.T))
            elif method == 'pca_cor':
                self.whiten_ = np.dot(np.dot(p_inv, G.T), V_inv)
                self.dewhiten_ = np.dot(V_sqrt, np.dot(G, p))
        else:
            raise Exception('Whitening method not found.')

        return self


    def transform(self, X, y=None, copy=None):
        """Perform whitening

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the whitening transform and rotate back
        to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_.T) + self.mean_
