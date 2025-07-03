"""Provide a subclass of ``mrg32k3a`` as an implementation of mrg32k3a with vector support."""

import numpy as np
import numba
import math

from mrg32k3a.mrg32k3a import MRG32k3a

# =========================================================================
# 1. Numba-JIT accelerated core helper functions
#    These are now specialized for each distribution to avoid typing errors.
# =========================================================================

# Constants used in mrg32k3a and in substream generation.
# P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random Number Generators'',
# Operations Research, 47, 1 (1999), 159--164.
# P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton,
# ``An Objected-Oriented Random-Number Package with Many Long Streams and Substreams'',
# Operations Research, 50, 6 (2002), 1073--1075.
# These are used in the core generator logic.
mrgm1 = 4294967087
mrgm2 = 4294944443
mrga12 = 1403580
mrga13n = 810728
mrga21 = 527612
mrga23n = 1370589


@numba.jit(nopython=True, cache=True)
def _mrg32k3a_numba_core(state):
    """
    Numba JIT-compiled version of the core MRG32k3a generator algorithm.

    This function advances the generator's state by one step and produces a
    single uniform random number in [0, 1). Using Numba's `nopython=True`
    mode provides a significant performance boost.

    Args:
        state (tuple): The current state of the generator, a tuple of six integers
                       (s1_0, s1_1, s1_2, s2_0, s2_1, s2_2).

    Returns:
        tuple: A tuple containing:
            - tuple: The next state of the generator.
            - float: A uniform random number in the interval [0, 1).
    """
    s1_0, s1_1, s1_2 = state[0], state[1], state[2]
    s2_0, s2_1, s2_2 = state[3], state[4], state[5]

    # First component of the generator
    p1 = (mrga12 * s1_1 - mrga13n * s1_0) % mrgm1
    # Second component of the generator
    p2 = (mrga21 * s2_2 - mrga23n * s2_0) % mrgm2

    # The new state is formed by shifting and including the new values
    next_state = (s1_1, s1_2, p1, s2_1, s2_2, p2)

    # Combination of the two components to produce the output
    z = (p1 - p2) % mrgm1
    if z > 0:
        u = z / (mrgm1 + 1.0)
    else:
        # Handle the case where (p1 - p2) is negative
        u = mrgm1 / (mrgm1 + 1.0)

    return next_state, u


@numba.jit(nopython=True, cache=True)
def _bsm_numba(u):
    """
    Numba JIT-compiled version of the Beasley-Springer-Moro algorithm.

    This function computes the inverse of the standard normal cumulative
    distribution function (CDF), also known as the normal quantile function.
    It efficiently transforms a uniform random number `u` into a standard
    normal random variate.

    Args:
        u (float): A uniform random number in the interval [0, 1).

    Returns:
        float: A standard normal random variate (mean=0, sigma=1).
    """
    # Constants are defined inside the function for Numba compatibility
    bsma = np.array([2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637], dtype=np.float64)
    bsmb = np.array([-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833], dtype=np.float64)
    bsmc = np.array([0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 0.0038405729373609,
                     0.0003951896511919, 0.0000321767881768, 0.0000002888167364, 0.0000003960315187])

    y = u - 0.5
    if abs(y) < 0.42:
        # Central region approximation
        r = y * y
        asum = bsma[0] + r * (bsma[1] + r * (bsma[2] + r * bsma[3]))
        bsum = 1.0 + r * (bsmb[0] + r * (bsmb[1] + r * (bsmb[2] + r * bsmb[3])))
        z = y * (asum / bsum)
    else:
        # Tail region approximation
        if y < 0:
            signum = -1.0
            r = u
        else:
            signum = 1.0
            r = 1.0 - u

        s = math.log(-math.log(r))
        t = bsmc[0] + s * (bsmc[1] + s * (bsmc[2] + s * (
                    bsmc[3] + s * (bsmc[4] + s * (bsmc[5] + s * (bsmc[6] + s * (bsmc[7] + s * bsmc[8])))))))
        z = signum * t
    return z


# --- Specialized Numba loops for each distribution ---

@numba.jit(nopython=True, cache=True)
def _uniform_loop(n_variates, initial_state):
    """
    Generates a batch of uniform random variates using the MRG32k3a core.

    Args:
        n_variates (int): The number of uniform variates to generate.
        initial_state (tuple): The starting state of the MRG generator.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` uniform random numbers.
    """
    variates = np.zeros(n_variates, dtype=np.float64)
    current_state = initial_state
    for i in range(n_variates):
        current_state, u = _mrg32k3a_numba_core(current_state)
        variates[i] = u
    return current_state, variates


@numba.jit(nopython=True, cache=True)
def _expovariate_loop(n_variates, initial_state, lambd):
    """
    Generates a batch of exponential random variates using the MRG32k3a core.

    Args:
        n_variates (int): The number of exponential variates to generate.
        initial_state (tuple): The starting state of the MRG generator.
        lambd (float): The rate parameter (lambda) of the exponential distribution.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` exponential random numbers.
    """
    variates = np.zeros(n_variates, dtype=np.float64)
    current_state = initial_state
    epsilon = 1e-15  # Add epsilon for numerical stability if u is close to 1.0
    for i in range(n_variates):
        current_state, u = _mrg32k3a_numba_core(current_state)
        # Inverse transform sampling for exponential distribution
        variates[i] = -math.log(1.0 - u + epsilon) / lambd
    return current_state, variates


@numba.jit(nopython=True, cache=True)
def _normalvariate_loop(n_variates, initial_state, mu, sigma):
    """
    Generates a batch of normal random variates using the MRG32k3a core.

    Args:
        n_variates (int): The number of normal variates to generate.
        initial_state (tuple): The starting state of the MRG generator.
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` normal random numbers.
    """
    variates = np.zeros(n_variates, dtype=np.float64)
    current_state = initial_state
    for i in range(n_variates):
        current_state, u = _mrg32k3a_numba_core(current_state)
        # Transform uniform `u` to standard normal, then scale and shift
        variates[i] = mu + sigma * _bsm_numba(u)
    return current_state, variates


@numba.jit(nopython=True, cache=True)
def _gumbelvariate_loop(n_variates, initial_state, mu, beta):
    """
    Generates a batch of Gumbel random variates using the MRG32k3a core.

    Args:
        n_variates (int): The number of Gumbel variates to generate.
        initial_state (tuple): The starting state of the MRG generator.
        mu (float): The location parameter of the Gumbel distribution.
        beta (float): The scale parameter of the Gumbel distribution.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` Gumbel random numbers.
    """
    variates = np.zeros(n_variates, dtype=np.float64)
    current_state = initial_state
    epsilon = 1e-15  # Add epsilon for numerical stability if u is close to 0.0
    for i in range(n_variates):
        current_state, u = _mrg32k3a_numba_core(current_state)
        # Inverse transform sampling for Gumbel distribution
        variates[i] = mu - beta * math.log(-math.log(u + epsilon))
    return current_state, variates


@numba.jit(nopython=True, cache=True)
def _poisson_numba_loop(n_variates, initial_state, lmbda):
    """
    Generates a batch of Poisson random variates using the MRG32k3a core.

    It uses two different algorithms based on the value of lambda (`lmbda`):
    - For lmbda < 35: Knuth's algorithm (multiplication of uniform randoms).
    - For lmbda >= 35: A normal approximation for higher efficiency.

    Args:
        n_variates (int): The number of Poisson variates to generate.
        initial_state (tuple): The starting state of the MRG generator.
        lmbda (float): The average rate (lambda) of the Poisson distribution.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` Poisson random integers.
    """
    variates = np.zeros(n_variates, dtype=np.int64)
    current_state = initial_state
    for i in range(n_variates):
        if lmbda < 35.0:
            # Knuth's algorithm for smaller lambda
            n = 0
            current_state, p = _mrg32k3a_numba_core(current_state)
            threshold = math.exp(-lmbda)
            while p >= threshold:
                current_state, u = _mrg32k3a_numba_core(current_state)
                p = p * u
                n += 1
            variates[i] = n
        else:
            # Normal approximation for larger lambda for better performance
            current_state, u_norm = _mrg32k3a_numba_core(current_state)
            z = _bsm_numba(u_norm)
            # Apply continuity correction
            variates[i] = max(math.ceil(lmbda + math.sqrt(lmbda) * z - 0.5), 0)
    return current_state, variates


@numba.jit(nopython=True, cache=True)
def _binomial_numba_loop(n_variates, initial_state, n, p):
    """
    Generates a batch of Binomial random variates using the MRG32k3a core.

    This implementation simulates `n` Bernoulli trials for each variate.

    Args:
        n_variates (int): The number of Binomial variates to generate.
        initial_state (tuple): The starting state of the MRG generator.
        n (int): The number of trials.
        p (float): The probability of success for each trial.

    Returns:
        tuple: A tuple containing:
            - tuple: The final state of the generator after generation.
            - np.ndarray: An array of `n_variates` Binomial random integers.
    """
    variates = np.zeros(n_variates, dtype=np.int64)
    current_state = initial_state
    for i in range(n_variates):
        successes = 0
        for _ in range(n):
            current_state, u = _mrg32k3a_numba_core(current_state)
            if u < p:
                successes += 1
        variates[i] = successes
    return current_state, variates


# =========================================================================
# 2. The new, fully accelerated inherited class
# =========================================================================

class MRG32k3a_numba(MRG32k3a):
    """
    An accelerated version of the MRG32k3a generator.

    This class inherits from the base `MRG32k3a` generator and provides
    high-performance "batch" generation methods for creating arrays of random
    variates. The core loops are JIT-compiled with Numba for maximum speed,
    making this class suitable for large-scale numerical simulations.

    The generator's state is properly managed, ensuring that subsequent calls
    to any batch method will continue the random number sequence without
    repetition.
    """

    def __init__(self, ref_seed=(12345, 12345, 12345, 12345, 12345, 12345), s_ss_sss_index=None):
        """
        Initializes the Numba-accelerated MRG32k3a generator.

        Args:
            ref_seed (tuple, optional): The starting seed for the generator,
                consisting of six integers. Defaults to a standard value.
            s_ss_sss_index (int, optional): The index of the stream/substream to use.
                Typically used for parallel computations. Defaults to None.
        """
        super().__init__(ref_seed, s_ss_sss_index)

    def _generate_batch(self, size, loop_function, params=(), dtype=np.float64):
        """
        Internal batch generation method that handles state management.

        This is a generic helper that calls a specified Numba-compiled loop
        function to generate variates and then updates the generator's state.

        Args:
            size (int): The number of variates to generate.
            loop_function (numba.jitted function): The Numba-compiled function
                that generates the random variates.
            params (tuple, optional): A tuple of additional parameters to pass
                to the loop function (e.g., mu, sigma). Defaults to ().
            dtype (np.dtype, optional): The data type of the output array.
                Defaults to np.float64.

        Returns:
            np.ndarray: An array of random variates of the specified size.
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("Size must be a non-negative integer.")
        if size == 0:
            return np.array([], dtype=dtype)

        initial_state = self.get_current_state()
        final_state, variates = loop_function(size, initial_state, *params)
        # CRITICAL: Update state to ensure the sequence continues from where it left off.
        self.seed(final_state)
        return variates

    def random_batch(self, size):
        """
        Generates a batch of uniform random numbers in the interval [0, 1).

        Args:
            size (int): The number of uniform random numbers to generate.

        Returns:
            np.ndarray: An array of `size` uniform random numbers.
        """
        return self._generate_batch(size, _uniform_loop)

    def normalvariate_batch(self, mu, sigma, size):
        """
        Generates a batch of normally distributed random variates.

        Args:
            mu (float): The mean of the normal distribution.
            sigma (float): The standard deviation of the normal distribution.
            size (int): The number of normal variates to generate.

        Returns:
            np.ndarray: An array of `size` normal random variates.
        """
        return self._generate_batch(size, _normalvariate_loop, params=(mu, sigma))

    def expovariate_batch(self, lambd, size):
        """
        Generates a batch of exponentially distributed random variates.

        Args:
            lambd (float): The rate parameter (lambda) of the distribution.
            size (int): The number of exponential variates to generate.

        Returns:
            np.ndarray: An array of `size` exponential random variates.
        """
        return self._generate_batch(size, _expovariate_loop, params=(lambd,))

    def gumbelvariate_batch(self, mu, beta, size):
        """
        Generates a batch of Gumbel-distributed random variates.

        Args:
            mu (float): The location parameter of the distribution.
            beta (float): The scale parameter of the distribution.
            size (int): The number of Gumbel variates to generate.

        Returns:
            np.ndarray: An array of `size` Gumbel random variates.
        """
        return self._generate_batch(size, _gumbelvariate_loop, params=(mu, beta))

    def lognormalvariate_batch(self, lq, uq, size):
        """
        Generates a batch of log-normally distributed random variates.

        The distribution is parameterized by the lower and upper quantiles
        corresponding to a 95% confidence interval.

        Args:
            lq (float): The lower quantile (e.g., 2.5th percentile).
            uq (float): The upper quantile (e.g., 97.5th percentile).
            size (int): The number of log-normal variates to generate.

        Returns:
            np.ndarray: An array of `size` log-normal random variates.
        """
        # Convert quantiles to the mean and sigma of the underlying normal distribution
        mu = (math.log(lq) + math.log(uq)) / 2
        sigma = (math.log(uq) - mu) / 1.96
        normal_variates = self.normalvariate_batch(mu, sigma, size)
        return np.exp(normal_variates)

    def poissonvariate_batch(self, lmbda, size):
        """
        Generates a batch of Poisson-distributed random variates.

        Args:
            lmbda (float): The average rate (lambda) of the distribution.
            size (int): The number of Poisson variates to generate.

        Returns:
            np.ndarray: An array of `size` Poisson random integers.
        """
        return self._generate_batch(size, _poisson_numba_loop, params=(lmbda,), dtype=np.int64)

    def binomialvariate_batch(self, n, p, size):
        """
        Generates a batch of Binomially-distributed random variates.

        Args:
            n (int): The number of trials for each variate.
            p (float): The probability of success for each trial.
            size (int): The number of Binomial variates to generate.

        Returns:
            np.ndarray: An array of `size` Binomial random integers.
        """
        return self._generate_batch(size, _binomial_numba_loop, params=(n, p), dtype=np.int64)

    def mvnormalvariate_batch(self, mean_vec, cov, size, factorized=False):
        """
        Generates a batch of multivariate normal random vectors.

        Args:
            mean_vec (array_like): The mean vector of the distribution.
            cov (array_like): The covariance matrix or its Cholesky factor.
            size (int): The number of random vectors to generate.
            factorized (bool, optional): If True, `cov` is assumed to be the
                pre-computed Cholesky factor of the covariance matrix.
                Defaults to False.

        Returns:
            np.ndarray: A `(size, n_cols)` array of multivariate normal
                random vectors.
        """
        n_cols = len(cov)
        if not factorized:
            # Perform Cholesky decomposition if the full covariance matrix is provided
            chol = np.linalg.cholesky(cov)
        else:
            # Use the provided factor directly
            chol = cov

        # Generate the required number of standard normal variates
        total_normal_variates = size * n_cols
        normal_variates = self.normalvariate_batch(0, 1, total_normal_variates).reshape(size, n_cols)

        mean_vec_arr = np.array(mean_vec)
        # Transform standard normal variates to the desired multivariate distribution
        # using the Cholesky factor and add the mean vector.
        return (chol @ normal_variates.T).T + mean_vec_arr