import numpy as np
import abc
import matplotlib.pyplot as plt


class AbstaractBaseFilter:
    """
        Abstract base class of adaptive filter

        Args
        ----------
        M : int
            Length of FIR filter weights vector (default = 10)
        step : float
            Step size of the algorithm, must be non-negative.
        leak : float
            Leakage factor, must be equal to or greater than zero and smaller than
            one. When greater than zero a leaky filter is used.
        eps : float
            Regularization factor for normalization procedure
        init_coeffs : array-like
            Initial FIR filter weights. Must match M. if None, filled with zeros
    """
    def __init__(self, M=10, step=0.1, leak=0.0, eps=0.001, init_coeffs=None):
        if type(M) == int:
            self.M = M
        else:
            raise ValueError('Length of FIR filter weights vector must be an integer')
        if type(step) == float and step > 0:
            self.step = step
        else:
            raise ValueError('Step size of the algorithm must be non-negative integer')
        if type(leak) == float and leak >= 0 and leak < 1:
            self.leak_step = (1 - step * leak)
        else:
            raise ValueError('Leakage factort be an integer between 0 and 1')
        if type(eps) == float and eps > 0:
            self.eps = eps
        else:
            raise ValueError('Regularization factor must be non-negative integer')
        if init_coeffs is None:
            self.init_coeffs = np.zeros(M)
        elif len(init_coeffs) == M:
            self.init_coeffs = init_coeffs
        else:
            raise ValueError('Length of FIR filter weights vector must be equal to M')
        pass

    @abc.abstractclassmethod
    def run(self, x, d,):
        pass

class LMSFilter(AbstaractBaseFilter):
    """
        LMS adaptive filter

        Args
        ----------
        M : int
            Length of FIR filter weights vector (default = 10)
        step : float
            Step size of the algorithm, must be non-negative.
        eps : float
            Regularization factor for normalization procedure.
        leak : float
            Leakage factor, must be equal to or greater than zero and smaller than
            one. When greater than zero a leaky filter is used.
        init_coeffs : array-like
            Initial FIR filter weights. Must match M. if None, filled with zeros
        norm: boolean
            Normalized/non-normalized version
    """
    def __init__(self, M=10, step=0.1, leak=0.0, eps=0.001, init_coeffs=None, norm=True):
        super(LMSFilter, self).__init__(M=M, step=step, leak=leak, init_coeffs=init_coeffs, eps=eps)
        self.norm = norm

    def run(self, x, d):
        """
        Performs LMS filtering

        Args
        ----------
        x : array-like
            One-dimensional filter input.
        d : array-like
            One-dimensional desired signal

        Output
        -------
        y : numpy.array
            Filtered output
        e : numpy.array
            Difference between output and desired signal
        w : numpy.matrix
            filter weights (history)

        """
        N = len(x)-self.M+1
        y = np.zeros(N)  # Filtered output
        e = np.zeros(N)  # Difference between output and desired signal
        w = self.init_coeffs  # Initial filter weights

        W = np.zeros((N, self.M))

        # Perform filtering
        for n in range(N):
            x_tmp = np.flipud(x[n:n+self.M])  # M latest datapoints (memory of filter)
            y[n] = np.dot(x_tmp, w)
            e[n] = d[n+self.M-1] - y[n]

            if self.norm:
                norm_factor = 1./(np.dot(x_tmp, x_tmp) + self.eps)
            else:
                norm_factor = 1

            w = self.leak_step * w + self.step * norm_factor * x_tmp * e[n]
            y[n] = np.dot(x_tmp, w)

            W[n] = w

        return y, e, W

class AffineProjectionFilter(AbstaractBaseFilter):
    """
            The affine projection algorithm (APA), is an adaptive scheme that estimates an unknown system based on
         multiple input vectors [1]. It is designed to improve the performance of other adaptive algorithms,
         mainly those that are LMS-based. The affine projection algorithm reuses old data resulting in fast convergence
         when the input signal is highly correlated, leading to a family of algorithms that can make trade-offs between
         computation complexity with convergence speed
         Paulo S. R. Diniz, Adaptive Filtering: Algorithms and Practical Implementation, Second Edition. Boston: Kluwer Academic Publishers, 2002

        Args
        ----------
        M : int
            Length of FIR filter weights vector (default = 10)
        step : float
            Step size of the algorithm, must be non-negative.
        L : int
            Projection order, must be integer larger than zero.
        eps : float
            Regularization factor for normalization procedure
        leak : float
            Leakage factor, must be equal to or greater than zero and smaller than
            one. When greater than zero a leaky LMS filter is used.
        init_coeffs : array-like
            Initial FIR filter weights. Must match M. if None, filled with zeros.

    """
    def __init__(self, M=10, step=0.1, leak=0.0, L=5, eps=0.001, init_coeffs=None):
        super(AffineProjectionFilter, self).__init__(M=M, step=step, leak=leak, init_coeffs=init_coeffs)
        self.eps = eps
        self.L = L
        self.epsI = self.eps * np.identity(self.L)

    def run(self, x, d):
        """
        Performs affine projection filtering

        Args
        ----------
        x : array-like
            One-dimensional filter input.
        d : array-like
            One-dimensional desired signal

        Output
        -------
        y : numpy.array
            Filtered output
        e : numpy.array
            Difference between output and desired signal
        w : numpy.matrix
            filter weights (history)

        """
        N = len(x)-self.M-self.L+1
        y = np.zeros(N)  # Filtered output
        e = np.zeros(N)  # Difference between output and desired signal
        w = self.init_coeffs  # Initial filter weights

        epsI = self.eps * np.identity(self.L)  # Regularization matrix

        W = np.zeros((N, self.M))

        # Perform filtering
        for n in range(N):
            # Generate X matrix and D vector with current data
            X = np.zeros((self.M, self.L))
            for k in np.arange(self.L):
                X[:, (self.L - k - 1)] = x[n + k:n + self.M + k]
            X = np.flipud(X)
            D = np.flipud(d[n + self.M - 1:n + self.M + self.L - 1])

            # Filter
            y_tmp = np.dot(X.T, w)
            e_tmp = D - y_tmp

            y[n] = y_tmp[0]
            e[n] = e_tmp[0]

            norm_factor = np.linalg.inv(epsI + np.dot(X.T, X))

            w = self.leak_step * w + self.step * np.dot(X, np.dot(norm_factor, e_tmp))

            W[n] = w

        return y, e, W


def affine_projection_filter(main, ref, M = 200, step = 0.05, L = 5, leak=0.0):
    """
    AffineProjectionFilter

    :param main: - Raw main signal  - shape (samples,)
    :param ref: - Raw ref signal  - shape (samples,)
    :param M: - Length of FIR filter weights vector
    :param step: - Step size of the algorithm, must be non-negative.
    :param L:  - Projection order, must be integer larger than zero.
    :return:
    """

    filter = AffineProjectionFilter(M=M, step=step, L=L, leak=leak)
    y, e, w = filter.run(x = ref, d = main)

    return e

def lms_filter(main, ref, M = 200, step = 0.05,  leak=0.0, norm=True):
    """
    LMS filter

    :param main: - Raw main signal  - shape (samples,)
    :param ref: - Raw ref signal  - shape (samples,)
    :param M: - Length of FIR filter weights vector
    :param step: - Step size of the algorithm, must be non-negative.
    :return:
    """

    filter = LMSFilter(M=M, step=step, leak=leak, norm=norm)
    y, e, w = filter.run(x = ref, d = main)

    return e
