import gzip
import pickle

import numpy as np
from jax import numpy as jnp

MNIST_PATH = "datasets/mnist.pkl"
FASHION_PATH = "datasets/fashion_{}.gz"

def get_mnist(path = MNIST_PATH):
  """ Load Larochelle (binarised) MNIST data """
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

def get_fashion_mnist(path = FASHION_PATH):

  def get_dataset(type):
    with gzip.open(path.format(type + "_labels"), "rb") as f:
      labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(path.format(type + "_images"), "rb") as f:
      images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
      images = images.reshape(len(labels), 784)

    # Binarise images
    images = np.round(images / 255.)

    # Convert to JAX arrays
    labels = jnp.array(labels) 
    images = jnp.array(images)

    return { type+"_x": images, type+"_y": labels }

  return { **get_dataset("train"), **get_dataset("test") }

def get_batches(data, batch_size, smaller_data=False, smaller_size=1000):
  """
    Split train data into batches. Discard last batch if uneven for equal size
    arrays.
  """
  if smaller_data:
    data = data[:smaller_size]
  num_batches = len(data) // batch_size
  batches = [ data[i*batch_size:(i+1)*batch_size] for i in range(num_batches) ]
  return jnp.array(batches)
