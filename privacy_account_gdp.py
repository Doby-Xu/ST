# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
r"""Implements privacy accounting for Gaussian Differential Privacy.

Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).
"""

from operator import le
import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(epoch, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""

  t = epoch * n / batch_size
  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * norm.cdf(1.5 / noise_multi) +
      3 * norm.cdf(-0.5 / noise_multi) - 2)


def compute_mu_poisson(epoch, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""

  t = epoch * n / batch_size
  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n


def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return norm.cdf(-eps / mu +
                  mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""

  def f(x):
    """Reversely solve dual by matching delta."""
    # return np.log(norm.cdf(-x / mu + mu / 2) - delta) - x - np.log(norm.cdf(-x / mu - mu / 2))
    return delta_eps_mu(x,mu) - delta
  return optimize.root_scalar(f, bracket=[1e-8,709], method='brentq').root


def compute_eps_uniform(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform(epoch, noise_multi, n, batch_size), delta)


def compute_eps_poisson(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of Poisson subsampling."""

  return eps_from_mu(
      compute_mu_poisson(epoch, noise_multi, n, batch_size), delta)

def compute_sigma_uniform(epoch, eps, delta, n, batch_size):
    def f(x):
        return compute_eps_uniform(epoch, x, n, batch_size, delta) - eps
    left = 0.3
    right = 10000
    # print(f(left))
    # print(f(right))
    while (np.abs(right - left)> 1e-5):
        mid = (right + left) / 2
        if f(left) * f(mid) <= 0:
            right = mid
        elif f(right) * f(mid) <= 0:
            left = mid
        else:
          print(f(left))
          print(f(right))
          print('no solution')
          break
    return mid

if __name__ == "__main__":
    eps = compute_eps_uniform(60,1,202599,50,1e-6)
    print(eps)
  # N = 50000#199364
  # epoch = 5#30
  # batch_size = 32#128
  # delta = 1e-6
  # # sigma_list = np.linspace(0.475, 2.0, 17)
  # sigma_list = [1.0, 5, 10, 15, 20, 25, 30]
  # for sigma in sigma_list:
  #     eps = compute_eps_uniform(epoch, sigma, N, batch_size, delta)
  #     # print('sigma=', sigma)
  #     print( eps)
  #
  # eps_list = np.array([1,2,4,8,16,32, 64, 128, 256])
  # for eps in eps_list:
  #     sigma = compute_sigma_uniform(epoch, eps, delta, N, batch_size)
  #     print('sigma=', sigma)
