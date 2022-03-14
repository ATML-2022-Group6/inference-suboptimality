import pickle
import numpy as np

from jax import numpy as jnp

MNIST_PATH = "datasets/mnist.pkl"

# Load Larochelle (binarised) MNIST data
def get_mnist(path = MNIST_PATH) -> dict[str, jnp.ndarray]:
  with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")
    mnist = {
      "train_x": np.concatenate([data[0][0], data[1][0]]),
      "train_y": np.concatenate([data[0][1], data[1][1]]),
      "test_x": data[2][0],
      "test_y": data[2][1],
    }
    for k in mnist:
      mnist[k] = jnp.array(mnist[k])
    return mnist