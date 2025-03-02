import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def dsigmoid(x):
    return x * (1 - x)

# Hyperparameters
input_dim = 3  # Number of input features
hidden_dim = 2  # Number of LSTM units

# Initialize LSTM parameters
np.random.seed(42)
Wf = np.random.randn(hidden_dim, hidden_dim + input_dim)  # Forget gate weights
bf = np.zeros((hidden_dim, 1))  # Forget gate bias

Wi = np.random.randn(hidden_dim, hidden_dim + input_dim)  # Input gate weights
bi = np.zeros((hidden_dim, 1))  # Input gate bias

Wo = np.random.randn(hidden_dim, hidden_dim + input_dim)  # Output gate weights
bo = np.zeros((hidden_dim, 1))  # Output gate bias

Wc = np.random.randn(hidden_dim, hidden_dim + input_dim)  # Cell state weights
bc = np.zeros((hidden_dim, 1))  # Cell state bias

# Input sequence
x_t = np.random.randn(input_dim, 1)  # Example input
h_prev = np.zeros((hidden_dim, 1))  # Previous hidden state
c_prev = np.zeros((hidden_dim, 1))  # Previous cell state

# Concatenate input and previous hidden state
concat = np.vstack((h_prev, x_t))

# Forget gate
ft = sigmoid(np.dot(Wf, concat) + bf)

# Input gate
it = sigmoid(np.dot(Wi, concat) + bi)

# Candidate cell state
tanh_c = np.tanh(np.dot(Wc, concat) + bc)

# Current cell state
c_t = ft * c_prev + it * tanh_c

# Output gate
ot = sigmoid(np.dot(Wo, concat) + bo)

# Current hidden state
h_t = ot * np.tanh(c_t)

# Display results
print("Forget gate output:\n", ft)
print("Input gate output:\n", it)
print("Candidate cell state:\n", tanh_c)
print("Cell state:\n", c_t)
print("Output gate output:\n", ot)
print("Hidden state:\n", h_t)
