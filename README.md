# MRG32k3a-Numba
This package provides A high-performance Python implementation of the MRG32k3a pseudo-random number generator of of L'Ecuyer (1999) and L'Ecuyer et al. (2002). It extends the implementation used in [mrg32k3a](https://pypi.org/project/mrg32k3a) to generate batch random numbers, accelerated with Numba. 
[![PyPI version](https://badge.fury.io/py/mrg32k3a-numba.svg)](https://badge.fury.io/py/mrg32k3a-numba)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library is designed for scientific computing, Monte Carlo simulations, and other applications that require large volumes of high-quality random numbers from various statistical distributions. It provides a simple, NumPy-friendly interface for generating batches of random variates at high speed.

## Key Features

* **High Performance:** Core generation loops are Just-In-Time (JIT) compiled with Numba for C-like speed.
* **Proven Algorithm:** Implements Pierre L'Ecuyer's MRG32k3a generator, known for its excellent statistical properties and long period ($2^{191}$).
* **Batch Generation:** Optimized for generating large arrays of random numbers in a single call, which is significantly faster than generating them one by one.
* **State Management:** The generator's state is automatically updated, ensuring that sequences of random numbers are continuous and reproducible.
* **Wide Range of Distributions:** Supports uniform, normal, exponential, Gumbel, log-normal, Poisson, binomial, and multivariate normal distributions.
* **NumPy Integration:** Seamlessly returns results as NumPy arrays for easy integration into existing data science and machine learning workflows.

## Installation

You can install `mrg32k3a-numba` directly from PyPI using pip:

```bash
pip install mrg32k3a-numba
```

This package depends on `numpy` and `numba`, which will be automatically installed if they are not already present.

## Quick Start

Using `mrg32k3a-numba` is straightforward. First, instantiate the generator, then call the desired batch generation method.

```python
import numpy as np
from mrg32k3a_numba import MRG32k3a_numba

# 1. Initialize the generator
# You can provide a seed for reproducibility
rng = MRG32k3a_numba(ref_seed=(123, 456, 789, 101, 112, 131))

# 2. Generate a batch of 100 uniform random numbers
uniform_variates = rng.random_batch(size=100)
print(f"Generated {len(uniform_variates)} uniform variates.")
print(f"First 5: {uniform_variates[:5]}\n")

# 3. Generate a batch of normally distributed numbers
# The generator's state is automatically updated, so this call will
# produce the next numbers in the sequence.
normal_variates = rng.normalvariate_batch(mu=100, sigma=15, size=5)
print(f"Generated 5 normal variates (mean=100, std=15):")
print(normal_variates)
```

## API and Usage Examples

The main class is `MRG32k3a_numba`. All generation methods require a `size` argument specifying the number of variates to produce.

### Uniform Distribution
Generate floating-point numbers uniformly distributed in the interval $[0, 1)$.

```python
# Generate 10 uniform random numbers
uniforms = rng.random_batch(size=10)
```

### Normal Distribution
Generate variates from a normal (or Gaussian) distribution.

```python
# Generate 100 numbers from a normal distribution with mean=0 and std_dev=1
normals = rng.normalvariate_batch(mu=0, sigma=1, size=100)
```

### Exponential Distribution
Generate variates from an exponential distribution. `lambd` is the rate parameter ($1.0 / \text{mean}$).

```python
# Generate 100 numbers from an exponential distribution with a mean of 2.0
# lambd = 1.0 / 2.0 = 0.5
exponentials = rng.expovariate_batch(lambd=0.5, size=100)
```

### Poisson Distribution
Generate integer variates from a Poisson distribution. `lmbda` is the average number of events (the $\lambda$ parameter).

```python
# Generate 50 numbers from a Poisson distribution with lambda=10
poissons = rng.poissonvariate_batch(lmbda=10, size=50)
```

### Binomial Distribution
Generate integer variates from a binomial distribution.

```python
# Simulate 50 experiments, each with 20 trials and a success probability of 0.25
binomials = rng.binomialvariate_batch(n=20, p=0.25, size=50)
```

### Multivariate Normal Distribution
Generate vectors from a multivariate normal distribution.

```python
mean_vector = [0, 5]
covariance_matrix = [[1.0, 0.8], [0.8, 1.5]]

# Generate 1000 2D vectors
mv_normals = rng.mvnormalvariate_batch(
    mean_vec=mean_vector,
    cov=covariance_matrix,
    size=1000
)
# The output `mv_normals` will have a shape of (1000, 2)
```

## Dependencies

* [NumPy](https://numpy.org/): For numerical operations and array manipulation.
* [Numba](https://numba.pydata.org/): For JIT compilation and performance acceleration.
* [mrg32k3a](https://pypi.org/project/mrg32k3a): The base pure-Python implementation of the generator.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or would like to contribute code, please feel free to open an issue or submit a pull request on the project's GitHub repository.

## Acknowledgments

This implementation is based on the combined multiple recursive generator MRG32k3a proposed by Pierre L'Ecuyer in the following paper:

> L'Ecuyer, Pierre (1999). Good parameters and implementations for combined multiple recursive random number generators. Operations Research 47(1):159-164.
> L'Ecuyer, Pierre, Richard Simard, E Jack Chen, and W. David Kelton (2002). An object-oriented random number package with many long streams and substreams. Operations Research 50(6):1073-1075.
> Eckman DJ, Henderson SG, Shashaani S (2023) SimOpt: A testbed for simulation-optimization experiments.
INFORMS Journal on Computing 35(2):495–508.

The Beasley-Springer-Moro algorithm is used for the fast approximation of the normal quantile function.
