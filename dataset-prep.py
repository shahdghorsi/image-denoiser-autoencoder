from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Check the shape of the data
print("Training data shape:", x_train.shape)  # (50000, 32, 32, 3)
print("Test data shape:", x_test.shape)       # (10000, 32, 32, 3)
print("Training labels shape:", y_train.shape)  # (50000, 1)
print("Test labels shape:", y_test.shape)       # (10000, 1)
