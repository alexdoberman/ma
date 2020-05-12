import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh


def get_power_spectral_density_matrix(observation, mask=None):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    normalization = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)

    psd = np.einsum('...dt,...et->...de', mask * observation,
                    observation.conj())
    psd /= normalization
    return psd



def get_power_spectral_density_matrix2(observation):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """

    bins, sensors, frames = observation.shape
    psd                   = np.zeros((bins, sensors, sensors), dtype=np.complex) 

    for t in range(frames):
        for k in range(bins):
            D = observation[k,:,t]
            psd[k,:,:] =  psd[k,:,:] + np.outer(D, D.conj())

    psd = psd / frames
    return psd



def get_pca_vector(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.
    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    # Calculate eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
    # Find max eigenvals
    vals = np.argmax(eigenvals, axis=-1)
    # Select eigenvec for max eigenval
    beamforming_vector = np.array(
            [eigenvecs[i, :, vals[i]] for i in range(eigenvals.shape[0])])
    # Reconstruct original shape
    beamforming_vector = np.reshape(beamforming_vector, shape[:-1])

    return beamforming_vector


def get_mvdr_vector(atf_vector, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector.

    :param atf_vector: Acoustic transfer function vector
        with shape (..., bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    while atf_vector.ndim > noise_psd_matrix.ndim - 1:
        noise_psd_matrix = np.expand_dims(noise_psd_matrix, axis=0)

    # Make sure matrix is hermitian
    noise_psd_matrix = 0.5 * (
        noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2)))

    numerator = solve(noise_psd_matrix, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

#    return beamforming_vector, denominator
    return beamforming_vector





def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector


def blind_analytic_normalization_legacy(vector, noise_psd_matrix):
    bins, sensors = vector.shape
    normalization = np.zeros(bins)
    for f in range(bins):
        normalization[f] = np.abs(np.sqrt(np.dot(
                np.dot(np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                       noise_psd_matrix[f]), vector[f, :])))
        normalization[f] /= np.abs(np.dot(
                np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                vector[f, :]))

    return vector * normalization[:, np.newaxis]


def blind_analytic_normalization(vector, noise_psd_matrix, eps=0):
    """Reduces distortions in beamformed ouptput.
        
    :param vector: Beamforming vector
        with shape (..., sensors)
    :param noise_psd_matrix:
        with shape (..., sensors, sensors)
    :return: Scaled Deamforming vector
        with shape (..., sensors)
    
    >>> vector = np.random.normal(size=(5, 6)).view(np.complex128)
    >>> vector.shape
    (5, 3)
    >>> noise_psd_matrix = np.random.normal(size=(5, 3, 6)).view(np.complex128)
    >>> noise_psd_matrix = noise_psd_matrix + noise_psd_matrix.swapaxes(-2, -1)
    >>> noise_psd_matrix.shape
    (5, 3, 3)
    >>> w1 = blind_analytic_normalization_legacy(vector, noise_psd_matrix)
    >>> w2 = blind_analytic_normalization(vector, noise_psd_matrix)
    >>> np.testing.assert_allclose(w1, w2)
        
    """
    nominator = np.einsum(
        '...a,...ab,...bc,...c->...',
        vector.conj(), noise_psd_matrix, noise_psd_matrix, vector
    )
    nominator = np.abs(np.sqrt(nominator))

    denominator = np.einsum(
        '...a,...ab,...b->...', vector.conj(), noise_psd_matrix, vector
    )
    denominator = np.abs(denominator)

    normalization = nominator / (denominator + eps)
    return vector * normalization[..., np.newaxis]


def apply_beamforming_vector(vector, mix):
    return np.einsum('...a,...at->...t', vector.conj(), mix)


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         normalization=False):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)

    # Beamforming vector
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    output = apply_beamforming_vector(W_gev, mix)

    return output.T


def svd_matrix_inversion(X, r):
    """
    Invert matrix using SVD

    :param X: (..., N,N) - matrix to invert
    :param r:  - count of main components,  r >=1 and r <= N
    :return:
        X_inv - (..., N,N) invert matrix
    """

    assert X.shape[-1] == X.shape[-2], 'X.shape[-1] != X.shape[-2]'
    assert r >=1 and r <= X.shape[-1], 'r >=1 and r <= X.shape[-1]'

    u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    X_inv = np.einsum('...ji,...j,...kj->...ik', vh[..., 0:r, :].conj(), 1.0 / s[..., 0:r], u[..., :, 0:r].conj())
    return X_inv

def svd_matrix_inversion_ratio(X, r):
    """
    Invert matrix using SVD

    :param X: (..., N,N) - matrix to invert
    :param r:  - ratio, r >0 and r <= 1
    :return:
        X_inv - (..., N,N) invert matrix
    """

    assert X.shape[-1] == X.shape[-2], 'X.shape[-1] != X.shape[-2]'
    assert r > 0.0 and r <= 1.0, 'r >=1 and r <= X.shape[-1]'

    X_inv = np.zeros(X.shape, dtype=np.complex64)
    u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    for i in range(X.shape[0]):
        _s = s[i,:]
        _r = len(_s[_s > _s[0] * r])
        #print('i = {}, r = {}'.format(i,_r))
        #print(_s)
        X_inv[i,:,:] = vh[i, 0:_r, :].conj().T @ np.diag(1.0/s[i, 0:_r]) @ u[i, :, 0:_r].conj().T
    return X_inv

def svd_matrix_regularization(X, r):
    """
    Invert matrix using SVD

    :param X: (..., N,N) - matrix to invert
    :param r:  - ratio, r >0 and r <= 1
    :return:
        X_inv - (..., N,N) invert matrix
    """

    assert X.shape[-1] == X.shape[-2], 'X.shape[-1] != X.shape[-2]'
    assert r > 0.0 and r <= 1.0, 'r >=1 and r <= X.shape[-1]'

    X_inv = np.zeros(X.shape, dtype=np.complex64)
    u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    for i in range(X.shape[0]):
        _s = s[i,:]
        _r = len(_s)
        _s = _s + _s[0]*r

        X_inv[i,:,:] = vh[i, 0:_r, :].conj().T @ np.diag(1.0/s[i, 0:_r]) @ u[i, :, 0:_r].conj().T
    return X_inv



def get_mvdr_vector_svd(atf_vector, noise_psd_matrix, type_reg, r):
    """
    Returns the MVDR beamforming vector + SVD regularization.

    :param atf_vector: Acoustic transfer function vector
        with shape (..., bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :param type_reg: Type SVD regularization
        0 - count  based method
        1 - percent from max based method

    :param r:
        param for type

    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    while atf_vector.ndim > noise_psd_matrix.ndim - 1:
        noise_psd_matrix = np.expand_dims(noise_psd_matrix, axis=0)

    noise_psd_matrix_inv = None
    if type_reg ==0:
        noise_psd_matrix_inv = svd_matrix_inversion(noise_psd_matrix, r)
    elif type_reg == 1:
        noise_psd_matrix_inv = svd_matrix_inversion_ratio(noise_psd_matrix, r)
    elif type_reg == 2:
        noise_psd_matrix_inv = svd_matrix_regularization(noise_psd_matrix, r)
    else:
        assert False, 'type = {} unsupported'.format(type_reg)

    numerator = np.einsum('...ij,...j->...i', noise_psd_matrix_inv, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector
