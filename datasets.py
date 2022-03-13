import pickle
import numpy as np

MNIST_PATH = "datasets/mnist.pkl"

# Load Larochelle (binarised) MNIST data
def get_mnist(path = MNIST_PATH) -> dict[str, np.ndarray]:
  with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")
    return {
      "train_x": np.concatenate(data[0][0], data[1][0]),
      "train_y": np.concatenate(data[0][1], data[1][1]),
      "test_x": data[2][0],
      "test_y": data[2][1],
    }