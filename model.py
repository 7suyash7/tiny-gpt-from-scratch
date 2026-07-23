"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    chars = set(text)
    return sorted(chars)

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    return {char: idx for idx, char in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    return {idx: char for idx, char in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return "".join(decode_int(token_id, itos) for token_id in ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    exp_logits = array_exp(logits)
    return exp_logits / sum_all(exp_logits)

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    exp_result = array_exp(np.array([large_value]))
    naive_exp = float(exp_result[0])

    return {
        "naive_exp": naive_exp,
        "overflowed": bool(np.isinf(naive_exp))
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    max_logit = max_along_axis(logits, axis=0)
    shifted_logits = logits - max_logit
    exp_logits = array_exp(shifted_logits)
    return exp_logits / sum_all(exp_logits)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    row_maxes = max_along_axis(logits, axis=1)[:, None]
    shifted_logits = logits - row_maxes
    exp_logits = array_exp(shifted_logits)
    row_sums = sum_keepdims(exp_logits, axis=1)
    return exp_logits / row_sums

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if not isinstance(text_blob, str):
        raise TypeError("text blob must be a string")
    
    if len(text_blob) == 0:
        raise ValueError("text blob must not be empty")
    
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    token_ids = encode_string(text, stoi)
    return np.array(token_ids, dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    return (data[:split_idx], data[split_idx:])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return max(1, int(default_size))

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i:i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    return data[i + 1:i + 1 + block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    return rng.integers(
        low=0,
        high=data_len - block_size,
        size=batch_size,
    )

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    windows = [
        slice_x_at_offset(data, offset, block_size)
        for offset in offsets
    ]

    return np.stack(windows, axis=0)

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    windows = [
        slice_y_at_offset(data, offset, block_size)
        for offset in offsets
    ]

    return np.stack(windows, axis=0)

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    offsets = sample_random_batch_offsets(
        data_len=len(data),
        block_size=block_size,
        batch_size=batch_size,
        rng=rng,
    )

    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)

    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for t in range(len(data) - 1):
        current_token = data[t]
        next_token = data[t + 1]
        n_matrix[current_token, next_token] += 1
    
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    n_matrix = allocate_count_matrix(vocab_size)

    current_tokens = data[:-1]
    next_tokens = data[1:]

    np.add.at(n_matrix, (current_tokens, next_tokens), 1)

    return n_matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return sum_keepdims(n_matrix, axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    row_sums = row_sums_of_counts(n_matrix)
    return n_matrix / row_sums

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probabilities = p_matrix[current_id]
    return int(rng.choice(len(probabilities), p=probabilities))

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = np.empty(length, dtype=np.int64)
    sequence[0] = start_id

    for t in range(1, length):
        sequence[t] = sample_next_token(
            p_matrix,
            current_id=sequence[t - 1],
            rng=rng,
        )
    
    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    return "".join(decode_int(token_id ,itos) for token_id in ids)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    probability = index_element(p_matrix, current_id, next_id)
    log_probability = array_log(np.array([probability]))[0]

    return float(log_probability)

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    total_nll = 0.0

    for t in range(len(data) - 1):
        current_id = data[t]
        next_id = data[t + 1]

        log_prob = log_prob_of_pair(
            p_matrix,
            current_id,
            next_id,
        )

        total_nll -= log_prob
    
    return float(total_nll)

# Step 56 - average_nll
def average_nll(p_matrix, data):
    total_nll = sum_negative_log_probs(p_matrix, data)
    num_bigrams = len(data) - 1
    return float(total_nll / num_bigrams)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    one_hot = make_2d_zeros(len(ids), vocab_size)
    one_hot[np.arange(len(ids)), ids] = 1.0
    return one_hot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    vocab_size = w.shape[0]

    onehot = one_hot_encode_batch(ids, vocab_size)
    onehot_result = forward_logits_onehot(onehot, w)
    index_result = w[ids]

    return {
        "onehot_result": onehot_result,
        "index_result": index_result,
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    batch_indices = np.arange(len(targets))
    return probs[batch_indices, targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return -np.mean(log_probs)

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    return (
        "For each example, probs = softmax(logits) and "
        "L = -(1 / B) * sum(log(probs[i, targets[i]])). "
        "Differentiating softmax followed by cross-entropy gives "
        "dL/dlogits[i, j] = "
        "(probs[i, j] - 1[j == targets[i]]) / B. "
        "Therefore, dL/dlogits = (probs - onehot(targets)) / B."
    )

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    batch_size = probs.shape[0]

    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1.0
    dlogits /= batch_size

    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids.\n"
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros(
        (vocab_size, dlogits.shape[1]),
        dtype=np.float64,
    )
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)

    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(
        ids,
        dlogits,
        vocab_size=w.shape[0],
    )

    new_w = sgd_update_w(w, dw, learning_rate)

    return {
        "w": new_w,
        "loss": float(loss)
    }

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    loss_history = []
    rng = np.random.default_rng(0)

    for step in range(num_steps):
        X, Y = get_batch(
            data,
            block_size,
            batch_size,
            rng,
        )

        ids = X.reshape(-1)
        targets = Y.reshape(-1)

        result = run_one_training_step(
            w,
            ids,
            targets,
            learning_rate,
        )

        w = result["w"]

        if step % log_every == 0:
            loss_history.append(result["loss"])
    
    return {
        "w": w,
        "loss_history": loss_history,
    }

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    rng = np.random.default_rng(0)

    generated_ids = [int(start_id)]
    current_id = int(start_id)

    for _ in range(num_tokens):
        ids = np.array([current_id], dtype=np.int64)

        logits = forward_logits_lookup(w, ids)
        probs = logits_to_probs_rowwise(logits)

        next_id = int(
            rng.choice(w.shape[1], p=probs[0])
        )

        generated_ids.append(next_id)
        current_id = next_id
    
    return decode_ids(generated_ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    y = matmul(x, w)

    return {
        "y": y,
        "cache": {
            "x": x,
            "w": w,
        },
    }

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    return (
        "Y = X @ W\n"
        "Each weight receives contributions from every batch row.\n"
        "dL/dW = X.T @ dY\n"
        "shapes: X.T (D_in, N), dY (N, D_out) -> dL/dW (D_in, D_out)"
    )

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    w = cache["w"]
    return matmul(dy, transpose_matrix(w))

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    x = cache["x"]
    return matmul(transpose_matrix(x), dy)

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = vector_matrix_broadcast_add(x, b)

    return {
        "y": y,
        "cache": {
            "b_shape": b.shape,
        },
    }

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    db = sum_axis0(dy)
    return db.reshape(cache["b_shape"])

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    y = np.maximum(0, x)

    return {
        "y": y,
        "cache": {
            "x": x,
        },
    }

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    x = cache["x"]
    return dy * (x > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    return compute_dlogits(probs, targets)

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    feature_dim = x.shape[-1]
    return sum_keepdims(x, axis=-1) / feature_dim

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    centered = x - mean
    squared_deviations = centered ** 2
    feature_dim = x.shape[-1]

    return sum_keepdims(squared_deviations, axis=-1) / feature_dim

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)

    scaled = elementwise_multiply(x_hat, gamma)
    y = vector_matrix_broadcast_add(scaled, beta)

    return {
        "y": y,
        "cache": {
            "x": x,
            "x_hat": x_hat,
            "mean": mean,
            "var": var,
            "gamma": gamma,
            "eps": eps,
        },
    }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    feature_dim = dy.shape[-1]
    mean_dy = sum_keepdims(dy, axis=-1) / feature_dim
    return dy - mean_dy

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    var = cache["var"]
    eps = cache["eps"]

    std = np.sqrt(var + eps)
    return dy / std

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache["x"]
    x_hat = cache["x_hat"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]

    # Backward through y = gamma * x_hat + beta
    dx_hat = dy * gamma

    # gamma and beta are shared across every dimension except features.
    reduction_axes = tuple(range(dy.ndim - 1))
    dgamma = np.sum(dy * x_hat, axis=reduction_axes)
    dbeta = np.sum(dy, axis=reduction_axes)

    # Backward through LayerNorm normalisation
    feature_dim = x.shape[-1]
    inv_std = 1.0 / np.sqrt(var + eps)

    sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
    sum_dx_hat_x_hat = np.sum(
        dx_hat * x_hat,
        axis=-1,
        keepdims=True,
    )

    dx = (
        inv_std
        / feature_dim
        * (
            feature_dim * dx_hat
            - sum_dx_hat
            - x_hat * sum_dx_hat_x_hat
        )
    )

    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    return layernorm_backward_full(d_out, cache)

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    random_values = np.random.randn(vocab_size, d_model)
    return scale_w_small(random_values, scale)

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    out = embedding_matrix[token_ids]

    cache = {
        "token_ids": token_ids,
        "vocab_size": embedding_matrix.shape[0],
    }

    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    token_ids = cache["token_ids"]
    vocab_size = cache["vocab_size"]
    d_model = d_out.shape[-1]

    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)

    np.add.at(dE, token_ids, d_out)

    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    random_values = make_2d_random(
        block_size,
        d_model,
        seed=None,
    )

    return scale_w_small(random_values, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    d_model = positional_matrix.shape[1]
    return slice_subblock(
        positional_matrix,
        0,
        seq_len,
        0,
        d_model,
    )

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    d_token_emb = d_out
    d_pos_emb = sum_axis0(d_out)

    return {
        "d_token_emb": d_token_emb,
        "d_pos_emb": d_pos_emb,
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    """Create single-head query, key, and value projection matrices."""
    Wq = scale_w_small(
        make_2d_random(d_model, d_head, seed=0),
        scale,
    )
    Wk = scale_w_small(
        make_2d_random(d_model, d_head, seed=1),
        scale,
    )
    Wv = scale_w_small(
        make_2d_random(d_model, d_head, seed=2),
        scale,
    )

    return {
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv,
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    return np.matmul(x, w_q)

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    return matmul(x, w_k)

# Step 102 - compute_value
def compute_value(x, w_v):
    return matmul(x, w_v)

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    k_transposed = np.swapaxes(k, -1, -2)
    return matmul(q, k_transposed)

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    row_maxes = max_along_axis(masked_scores, axis=-1)[..., None]
    shifted_scores = masked_scores - row_maxes

    exp_scores = array_exp(shifted_scores)
    row_sums = sum_keepdims(exp_scores, axis=-1)

    return exp_scores / row_sums

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return matmul(attn, v)

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    return matmul(attn_out, w_o)

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache["attn_out"]
    w_o = cache["w_o"]

    # dX = dY @ W.T
    d_attn_out = matmul(d_proj, w_o.T)

    # Combine batch and sequence positions before computing dW.
    d_head = attn_out.shape[-1]
    d_model = d_proj.shape[-1]

    attn_flat = attn_out.reshape(-1, d_head)
    d_proj_flat = d_proj.reshape(-1, d_model)

    # dW = X.T @ dY
    dw_o = matmul(attn_flat.T, d_proj_flat)

    return {
        "d_attn_out": d_attn_out,
        "dw_o": dw_o,
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    attn = cache["attn"]
    v = cache["v"]

    v_t = np.swapaxes(v, -1, -2)
    attn_t = np.swapaxes(attn, -1, -2)

    d_attn = matmul(d_attn_out, v_t)
    d_v = matmul(attn_t, d_attn_out)

    return {
        "d_attn": d_attn,
        "d_v": d_v,
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache["attn"]
    causal_mask = cache["causal_mask"]

    weighted_sum = np.sum(
        d_attn * attn,
        axis=-1,
        keepdims=True,
    )

    d_masked_scores = attn * (d_attn - weighted_sum)

    return np.where(
        causal_mask,
        d_masked_scores,
        0.0,
    )

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache["q"]
    k = cache["k"]

    d_q = matmul(d_scores, k)

    d_scores_t = np.swapaxes(d_scores, -1, -2)
    d_k = matmul(d_scores_t, q)

    return {
        "d_q": d_q,
        "d_k": d_k,
    }

# Step 115 - qkv_projection_backward
import numpy as np

def qkv_projection_backward(d_q, d_k, d_v, cache):
    """Backprop through Q=x@Wq, K=x@Wk, V=x@Wv."""
    x = cache["x"]
    w_q = cache["w_q"]
    w_k = cache["w_k"]
    w_v = cache["w_v"]

    # Gradients flowing back into the shared input x.
    dx_q = matmul(d_q, np.swapaxes(w_q, -1, -2))
    dx_k = matmul(d_k, np.swapaxes(w_k, -1, -2))
    dx_v = matmul(d_v, np.swapaxes(w_v, -1, -2))

    dx = dx_q + dx_k + dx_v

    # Flatten batch and time so weight gradients accumulate over both.
    d_model = x.shape[-1]
    d_head = d_q.shape[-1]

    x_flat = x.reshape(-1, d_model)
    d_q_flat = d_q.reshape(-1, d_head)
    d_k_flat = d_k.reshape(-1, d_head)
    d_v_flat = d_v.reshape(-1, d_head)

    dw_q = matmul(x_flat.T, d_q_flat)
    dw_k = matmul(x_flat.T, d_k_flat)
    dw_v = matmul(x_flat.T, d_v_flat)

    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v,
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    
    if d_model % n_heads != 0:
        raise ValueError("n_heads must evenly divide d_model")
    
    return {
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_model": d_model,
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv for multi-head attention."""
    Wq = scale_w_small(
        make_2d_random(d_model, d_model, seed=0),
        scale,
    )
    Wk = scale_w_small(
        make_2d_random(d_model, d_model, seed=1),
        scale,
    )
    Wv = scale_w_small(
        make_2d_random(d_model, d_model, seed=2),
        scale,
    )

    return {
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv,
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    random_values = make_2d_random(
        d_model,
        d_model,
        seed=0,
    )

    return scale_w_small(random_values, scale)

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    batch_size, seq_len, _ = x.shape
    return x.reshape(batch_size, seq_len, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    transposed = np.transpose(x_heads, (0, 2, 1, 3))
    return np.ascontiguousarray(transposed)

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    shape = get_array_shape(x)
    return shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    """Return the per-head dimension for multi-head attention."""
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")

    if d_model % n_heads != 0:
        raise ValueError("n_heads must evenly divide d_model")

    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    masked_scores = apply_causal_mask(scores, mask)

    original_shape = masked_scores.shape
    seq_len = original_shape[-1]

    flattened_scores = masked_scores.reshape(-1, seq_len)
    flattened_weights = stable_softmax_2d_rowwise(flattened_scores)

    return flattened_weights.reshape(original_shape)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    return np.matmul(weights, v_heads)

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    transposed = np.transpose(x_heads, (0, 2, 1, 3))
    return np.ascontiguousarray(transposed)

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    shape = get_array_shape(x_heads_back)
    return int(shape[1])

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    batch_size, seq_len, n_heads, d_head = x_heads_back.shape
    d_model = n_heads * d_head

    return x_heads_back.reshape(batch_size, seq_len, d_model)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    linear_result = linear_forward(merged, w_out)
    bias_result = bias_add_forward(linear_result["y"], b_out)

    return {
      "out": bias_result["y"],
      "cache": {
        "merged": merged,
        "w_out": w_out,
      },
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Recover gradients with shape (B, n_heads, T, d_head)."""
    n_heads = shape_info["n_heads"]
    d_head = shape_info["d_head"]

    # (B, T, d_model) -> (B, T, n_heads, d_head)
    d_heads_back = reshape_to_heads(
        d_merged,
        n_heads,
        d_head,
    )

    # (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
    return transpose_heads_to_front(d_heads_back)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    linear_result = linear_forward(x, w1)
    bias_result = bias_add_forward(linear_result["y"], b1)

    return {
        "h1": bias_result["y"],
        "cache": {
            "x": x,
            "w1": w1,
        },
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    relu_result = relu_forward(h1)

    a1 = relu_result["y"]
    cache = {
        "h1": h1,
    }

    return a1, cache

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    linear_result = linear_forward(a1, w2)
    bias_result = bias_add_forward(linear_result["y"], b2)

    return {
        "h2": bias_result["y"],
        "cache": {
            "a1": a1,
            "w2": w2,
        },
    }

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN."""
    x = cache["x"]
    w1 = cache["w1"]
    h1 = cache["h1"]
    a1 = cache["a1"]
    w2 = cache["w2"]

    original_x_shape = x.shape

    # Collapse batch and time into one dimension.
    x_flat = x.reshape(-1, x.shape[-1])
    h1_flat = h1.reshape(-1, h1.shape[-1])
    a1_flat = a1.reshape(-1, a1.shape[-1])
    d_out_flat = d_out.reshape(-1, d_out.shape[-1])

    # Backward through y = a1 @ w2 + b2.
    linear2_cache = {
        "x": a1_flat,
        "w": w2,
    }

    d_a1_flat = linear_backward_dx(d_out_flat, linear2_cache)
    dw2 = linear_backward_dw(d_out_flat, linear2_cache)
    db2 = bias_add_backward_db(
        d_out_flat,
        {"b_shape": (w2.shape[1],)},
    )

    # Backward through a1 = ReLU(h1).
    d_h1_flat = relu_backward(
        d_a1_flat,
        {"x": h1_flat},
    )

    # Backward through h1 = x @ w1 + b1.
    linear1_cache = {
        "x": x_flat,
        "w": w1,
    }

    dx_flat = linear_backward_dx(d_h1_flat, linear1_cache)
    dw1 = linear_backward_dw(d_h1_flat, linear1_cache)
    db1 = bias_add_backward_db(
        d_h1_flat,
        {"b_shape": (w1.shape[1],)},
    )

    dx = dx_flat.reshape(original_x_shape)

    return {
        "dx": dx,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    d_x = d_y.copy()
    d_sublayer_out = d_y.copy()

    return d_x, d_sublayer_out

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    """Compute y = x + Sublayer(LayerNorm(x))."""
    ln_result = layernorm_forward_affine(
        x,
        ln_params["gamma"],
        ln_params["beta"],
        ln_params.get("eps", 1e-5),
    )

    x_norm = ln_result["y"]

    sublayer_result = sublayer_fn(
        x_norm,
        sublayer_params,
    )

    y = residual_forward(
        x,
        sublayer_result["y"],
    )

    return {
        "y": y,
        "cache": {
            "x": x,
            "ln_cache": ln_result["cache"],
            "sublayer_cache": sublayer_result["cache"],
        },
    }

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward."""

    def attention_sublayer(x_norm, attn_params):
        """Run causal multi-head self-attention."""

        w_q = attn_params["Wq"]
        w_k = attn_params["Wk"]
        w_v = attn_params["Wv"]
        w_o = attn_params["Wo"]
        b_o = attn_params["bo"]
        n_heads = attn_params["n_heads"]

        B, T, d_model = x_norm.shape
        d_head = compute_d_head(d_model, n_heads)

        # Project input into Q, K and V.
        q = compute_query(x_norm, w_q)
        k = compute_key(x_norm, w_k)
        v = compute_value(x_norm, w_v)

        # (B, T, d_model) -> (B, n_heads, T, d_head)
        q_heads = transpose_heads_to_front(
            reshape_to_heads(q, n_heads, d_head)
        )
        k_heads = transpose_heads_to_front(
            reshape_to_heads(k, n_heads, d_head)
        )
        v_heads = transpose_heads_to_front(
            reshape_to_heads(v, n_heads, d_head)
        )

        # Scaled causal self-attention.
        scores = compute_attention_scores(q_heads, k_heads)
        scaled_scores = scale_attention_scores(scores, d_head)

        causal_mask = build_causal_mask(T)
        weights = multihead_masked_softmax_scores(
            scaled_scores,
            causal_mask,
        )

        head_out = multihead_weighted_sum(weights, v_heads)

        # (B, n_heads, T, d_head) -> (B, T, d_model)
        heads_back = transpose_heads_to_back(head_out)
        merged = merge_heads_to_d_model(heads_back)

        projection = multihead_output_projection_forward(
            merged,
            w_o,
            b_o,
        )

        return {
            "y": projection["out"],
            "cache": {
                "x": x_norm,

                # Attention activations.
                "q": q_heads,
                "k": k_heads,
                "v": v_heads,
                "attn": weights,
                "causal_mask": causal_mask,
                "attn_out": head_out,
                "merged": merged,

                # Parameters needed by backward.
                "w_q": w_q,
                "w_k": w_k,
                "w_v": w_v,
                "w_o": w_o,

                # Shape bookkeeping.
                "shape_info": {
                    "B": B,
                    "T": T,
                    "n_heads": n_heads,
                    "d_head": d_head,
                },
            },
        }

    def ffn_sublayer(x_norm, ffn_params):
        """Run the two-layer position-wise FFN."""

        first = ffn_linear_one_forward(
            x_norm,
            ffn_params["w1"],
            ffn_params["b1"],
        )
        h1 = first["h1"]

        a1, _ = ffn_activation_forward(h1)

        second = ffn_linear_two_forward(
            a1,
            ffn_params["w2"],
            ffn_params["b2"],
        )

        return {
            "y": second["h2"],
            "cache": {
                "x": x_norm,
                "w1": ffn_params["w1"],
                "h1": h1,
                "a1": a1,
                "w2": ffn_params["w2"],
            },
        }

    # y1 = x + Attention(LN1(x))
    attn_branch = pre_layernorm_sublayer_forward(
        x,
        block_params["ln1"],
        attention_sublayer,
        block_params["attn"],
    )

    y1 = attn_branch["y"]

    # y2 = y1 + FFN(LN2(y1))
    ffn_branch = pre_layernorm_sublayer_forward(
        y1,
        block_params["ln2"],
        ffn_sublayer,
        block_params["ffn"],
    )

    return {
        "y": ffn_branch["y"],
        "cache": {
            "attn_branch": attn_branch["cache"],
            "ffn_branch": ffn_branch["cache"],
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block."""

    # Rebuild a guaranteed-complete cache from the original block input.
    x = cache["attn_branch"]["x"]
    complete_cache = _complete_block_cache(x, block_params)

    attn_branch = complete_cache["attn_branch"]
    ffn_branch = complete_cache["ffn_branch"]

    # ---------------------------------------------------------
    # 1. Backprop through:
    # y = h1 + FFN(LN2(h1))
    # ---------------------------------------------------------

    # Residual add sends d_y down both branches.
    d_h1_skip, d_ffn_out = residual_backward(d_y)

    # Backward through FFN.
    d_ln2_out, ffn_grads = _ffn_sublayer_backward(
        d_ffn_out,
        ffn_branch["sublayer_cache"],
        block_params["ffn"],
    )

    # Backward through LN2.
    d_h1_norm, d_ln2_gamma, d_ln2_beta = layernorm_backward_affine(
        d_ln2_out,
        ffn_branch["ln_cache"],
    )

    # Add the residual and transformed-path gradients.
    d_h1 = d_h1_skip + d_h1_norm

    # ---------------------------------------------------------
    # 2. Backprop through:
    # h1 = x + Attn(LN1(x))
    # ---------------------------------------------------------

    # Residual add again sends the gradient down both paths.
    d_x_skip, d_attn_out = residual_backward(d_h1)

    # Backward through attention.
    d_ln1_out, attn_grads = _attn_sublayer_backward(
        d_attn_out,
        attn_branch["sublayer_cache"],
        block_params["attn"],
    )

    # Backward through LN1.
    d_x_norm, d_ln1_gamma, d_ln1_beta = layernorm_backward_affine(
        d_ln1_out,
        attn_branch["ln_cache"],
    )

    # Final gradient with respect to the block input.
    d_x = d_x_skip + d_x_norm

    grads = {
        "ln1": {
            "gamma": d_ln1_gamma,
            "beta": d_ln1_beta,
        },
        "ln2": {
            "gamma": d_ln2_gamma,
            "beta": d_ln2_beta,
        },
        "attn": attn_grads,
        "ffn": ffn_grads,
    }

    return d_x, grads

# Step 140 - stack_transformer_blocks
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts."""
    choose_attention_head_config(d_model, n_heads)

    blocks = []

    for _ in range(n_layers):
        qkv = create_multihead_qkv_projections(d_model)
        Wo = create_multihead_output_projection(d_model)

        W1 = scale_w_small(
            make_2d_random(d_model, d_ff, seed=3),
            0.02,
        )

        W2 = scale_w_small(
            make_2d_random(d_ff, d_model, seed=4),
            0.02,
        )

        blocks.append({
            "ln1": {
                "gamma": np.ones(d_model),
                "beta": np.zeros(d_model),
            },
            "attn": {
                "Wq": qkv["Wq"],
                "Wk": qkv["Wk"],
                "Wv": qkv["Wv"],
                "Wo": Wo,
                "bo": np.zeros(d_model),
            },
            "ln2": {
                "gamma": np.ones(d_model),
                "beta": np.zeros(d_model),
            },
            "ffn": {
                "W1": W1,
                "b1": np.zeros(d_ff),
                "W2": W2,
                "b2": np.zeros(d_model),
            },
        })

    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    h = x
    caches = []

    for block_params in blocks:
        result = transformer_block_forward(h, block_params)
        h = result["y"]
        caches.append(result["cache"])
    
    return h, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    d_x = d_y
    grads_list = [None] * len(blocks)

    for i in range(len(blocks) -1, -1, -1):
      d_x, block_grads = transformer_block_backward(
        d_x,
        caches[i],
        blocks[i],
      )

      grads_list[i] = block_grads
    
    return d_x, grads_list

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm independently at every token position."""
    B, T, d_model = x.shape

    # Treat every token position as an independent row.
    x_flat = x.reshape(B * T, d_model)

    ln_result = layernorm_forward_affine(
        x_flat,
        gamma,
        beta,
        eps=1e-5,
    )

    y = ln_result["y"].reshape(B, T, d_model)
    ln_cache = ln_result["cache"]

    cache = {
        "x": x,
        "mean": ln_cache["mean"].reshape(B, T, 1),
        "var": ln_cache["var"].reshape(B, T, 1),
        "x_hat": ln_cache["x_hat"].reshape(B, T, d_model),
        "gamma": gamma,
    }

    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B, T, d_model) to vocabulary logits."""
    linear_result = linear_forward(x, w_lm)
    bias_result = bias_add_forward(
        linear_result["y"],
        b_lm,
    )

    return {
        "logits": bias_result["y"],
        "cache": {
            "x": x,
            "w_lm": w_lm,
        },
    }

# Step 145 - full_model_forward
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head."""
    seq_len = x_ids.shape[1]

    # 1. Token embeddings: (B, T) -> (B, T, d_model)
    token_emb, token_cache = token_embedding_forward(
        x_ids,
        model_params["tok_emb"],
    )

    # 2. Positional embeddings: (block_size, d_model) -> (T, d_model)
    pos_emb = slice_positional_embedding(
        model_params["pos_emb"],
        seq_len,
    )

    # Broadcasting adds the same position vectors across the batch.
    h = add_token_and_positional_embeddings(
        token_emb,
        pos_emb,
    )

    # 3. Transformer block stack
    h, block_caches = forward_through_all_blocks(
        h,
        model_params["blocks"],
    )

    # 4. Final LayerNorm
    h_norm, ln_f_cache = final_layernorm_forward(
        h,
        model_params["ln_f"]["gamma"],
        model_params["ln_f"]["beta"],
    )

    # 5. Language-model head
    lm_result = lm_head_linear_forward(
        h_norm,
        model_params["lm_head"]["w_lm"],
        model_params["lm_head"]["b_lm"],
    )

    logits = lm_result["logits"]

    caches = {
        "emb": {
            "token_cache": token_cache,
            "seq_len": seq_len,
        },
        "blocks": block_caches,
        "ln_f": ln_f_cache,
        "lm_head": lm_result["cache"],
    }

    return logits, caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    """Backprop through LM head, final LN, blocks, and embeddings."""

    # ---------------------------------------------------------
    # 1. Language-model head backward
    # logits = ln_out @ w_lm + b_lm
    # ---------------------------------------------------------
    lm_cache = caches["lm_head"]
    lm_x = lm_cache["x"]
    w_lm = lm_cache["w_lm"]

    d_model = lm_x.shape[-1]
    vocab_size = d_logits.shape[-1]

    lm_x_flat = lm_x.reshape(-1, d_model)
    d_logits_flat = d_logits.reshape(-1, vocab_size)

    linear_cache = {
        "x": lm_x_flat,
        "w": w_lm,
    }

    d_ln_flat = linear_backward_dx(
        d_logits_flat,
        linear_cache,
    )
    d_w_lm = linear_backward_dw(
        d_logits_flat,
        linear_cache,
    )
    d_b_lm = bias_add_backward_db(
        d_logits_flat,
        {"b_shape": model_params["lm_head"]["b_lm"].shape},
    )

    d_ln = d_ln_flat.reshape(lm_x.shape)

    # ---------------------------------------------------------
    # 2. Final LayerNorm backward
    # ---------------------------------------------------------
    ln_cache = dict(caches["ln_f"])
    ln_cache["eps"] = 1e-5

    ln_grads = layernorm_backward_full(
        d_ln,
        ln_cache,
    )

    d_blocks_out = ln_grads["dx"]

    # ---------------------------------------------------------
    # 3. Transformer block stack backward
    # ---------------------------------------------------------
    d_emb, block_grads = backward_through_all_blocks(
        d_blocks_out,
        caches["blocks"],
        model_params["blocks"],
    )

    # ---------------------------------------------------------
    # 4. Token + positional embedding backward
    # ---------------------------------------------------------
    emb_grads = embedding_sum_backward(d_emb)

    d_token_values = emb_grads["d_token_emb"]
    d_pos_slice = emb_grads["d_pos_emb"]

    # The step specification calls this "tok_cache". The fallback supports
    # the earlier forward implementation that stored it as "token_cache".
    emb_cache = caches["emb"]
    tok_cache = emb_cache.get(
        "tok_cache",
        emb_cache.get("token_cache"),
    )

    d_tok_emb = token_embedding_backward(
        d_token_values,
        tok_cache,
    )

    seq_len = emb_cache["seq_len"]

    # Positional parameters may be larger than the current sequence.
    d_pos_emb = np.zeros_like(model_params["pos_emb"])
    d_pos_emb[:seq_len] = d_pos_slice

    return {
        "tok_emb": d_tok_emb,
        "pos_emb": d_pos_emb,
        "blocks": block_grads,
        "ln_f": {
            "gamma": ln_grads["dgamma"],
            "beta": ln_grads["dbeta"],
        },
        "lm_head": {
            "w_lm": d_w_lm,
            "b_lm": d_b_lm,
        },
    }

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam buffers matching the nested parameter tree."""

    def build_zero_tree(node):
        if isinstance(node, np.ndarray):
            return np.zeros_like(node)

        if isinstance(node, dict):
            return {
                key: build_zero_tree(value)
                for key, value in node.items()
            }

        if isinstance(node, list):
            return [
                build_zero_tree(value)
                for value in node
            ]

        # Preserve non-trainable metadata such as block_size or vocab_size.
        return node

    m = build_zero_tree(model_params)
    v = build_zero_tree(model_params)

    return m, v

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    # TODO: return the next Adam step counter value
    return int(t) + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    old_component = elementwise_multiply(m, beta1)
    new_component = elementwise_multiply(grad, 1.0 - beta1)

    return elementwise_add(old_component, new_component)

# Step 151 - adam_update_second_moment
def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    grad_squared = elementwise_multiply(grad, grad)

    old_component = elementwise_multiply(v_prev, beta2)
    new_component = elementwise_multiply(
        grad_squared,
        1.0 - beta2,
    )

    return elementwise_add(old_component, new_component)

# Step 152 - adam_bias_correction
def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    return m_hat, v_hat

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return param - update

# Step 154 - wire_full_training_loop
import numpy as np

def wire_full_training_loop(
    params,
    train_ids,
    val_ids,
    block_size,
    batch_size,
    n_steps,
    lr,
    betas,
    eps,
):
    """Run the full GPT training loop and return (updated_params, history)."""
    beta1, beta2 = betas

    rng = np.random.default_rng(0)
    m_tree, v_tree = initialize_adam_moments(params)
    t = initialize_adam_step_counter()

    history = []

    def adam_update_tree(param_node, grad_node, m_node, v_node, step):
        """Recursively update matching ndarray leaves."""

        if isinstance(param_node, np.ndarray):
            new_m = adam_update_first_moment(
                m_node,
                grad_node,
                beta1,
            )
            new_v = adam_update_second_moment(
                v_node,
                grad_node,
                beta2,
            )

            m_hat, v_hat = adam_bias_correction(
                new_m,
                new_v,
                beta1,
                beta2,
                step,
            )

            new_param = adam_parameter_update(
                param_node,
                m_hat,
                v_hat,
                lr,
                eps,
            )

            return new_param, new_m, new_v

        if isinstance(param_node, dict):
            new_params = {}
            new_m_tree = {}
            new_v_tree = {}

            for key, value in param_node.items():
                # Metadata such as block_size and vocab_size has no gradient.
                child_grad = (
                    grad_node.get(key)
                    if isinstance(grad_node, dict)
                    else None
                )

                new_value, new_m, new_v = adam_update_tree(
                    value,
                    child_grad,
                    m_node[key],
                    v_node[key],
                    step,
                )

                new_params[key] = new_value
                new_m_tree[key] = new_m
                new_v_tree[key] = new_v

            return new_params, new_m_tree, new_v_tree

        if isinstance(param_node, list):
            new_params = []
            new_m_tree = []
            new_v_tree = []

            for i, value in enumerate(param_node):
                child_grad = (
                    grad_node[i]
                    if isinstance(grad_node, list)
                    else None
                )

                new_value, new_m, new_v = adam_update_tree(
                    value,
                    child_grad,
                    m_node[i],
                    v_node[i],
                    step,
                )

                new_params.append(new_value)
                new_m_tree.append(new_m)
                new_v_tree.append(new_v)

            return new_params, new_m_tree, new_v_tree

        # Preserve non-array metadata unchanged.
        return param_node, m_node, v_node

    for step in range(n_steps):
        # 1. Sample training sequences and next-token targets.
        x_batch, y_batch = get_batch(
            train_ids,
            block_size,
            batch_size,
            rng,
        )

        # 2. Full GPT forward pass.
        logits, caches = full_model_forward(
            x_batch,
            params,
        )

        B, T, vocab_size = logits.shape

        # 3. Flatten positions so existing 2D CE helpers can be reused.
        logits_flat = logits.reshape(B * T, vocab_size)
        targets_flat = y_batch.reshape(B * T)

        probs_flat = logits_to_probs_rowwise(logits_flat)
        loss = cross_entropy_loss(
            probs_flat,
            targets_flat,
        )

        # 4. Cross-entropy gradient with respect to logits.
        d_logits_flat = softmax_cross_entropy_backward(
            probs_flat,
            targets_flat,
        )
        d_logits = d_logits_flat.reshape(logits.shape)

        # 5. Backpropagate through the complete GPT.
        grads = full_model_backward(
            d_logits,
            caches,
            params,
        )

        # 6. Adam increments before bias correction.
        t = adam_increment_step(t)

        # 7. Update every ndarray in the nested parameter tree.
        params, m_tree, v_tree = adam_update_tree(
            params,
            grads,
            m_tree,
            v_tree,
            t,
        )

        history.append({
            "step": step,
            "train_loss": float(loss),
        })

    return params, history

# Step 155 - logging_and_validation_loss
def logging_and_validation_loss(
    params,
    val_ids,
    block_size,
    batch_size,
    n_eval_batches,
):
    """Estimate validation loss over reproducible validation batches."""
    rng = np.random.default_rng(0)
    losses = []

    for _ in range(n_eval_batches):
        x_batch, y_batch = get_batch(
            val_ids,
            block_size,
            batch_size,
            rng,
        )

        logits, _ = full_model_forward(
            x_batch,
            params,
        )

        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y_batch.reshape(-1)

        probs = stable_softmax_2d_rowwise(logits_flat)
        loss = cross_entropy_loss(probs, targets_flat)

        losses.append(float(loss))

    return float(np.mean(losses))

# Step 156 - encode_prompt
import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    token_ids = encode_string(prompt, stoi)
    return np.array([token_ids], dtype=np.int64)

# Step 157 - crop_context_to_block_size
def crop_context_to_block_size(context_ids, block_size):
    if context_ids.shape[1] <= block_size:
        return context_ids
    
    return context_ids[:, -block_size:]

# Step 158 - forward_to_get_logits
def forward_to_get_logits(params, context_ids):
    """Run the full model forward and return only the logits tensor."""
    logits, _ = full_model_forward(
        context_ids,
        params,
    )

    return logits

# Step 159 - take_last_position_logits
def take_last_position_logits(logits):
    """Return logits at the final time step with shape (1, vocab_size)."""
    return logits[:, -1, :]

# Step 160 - apply_temperature
def apply_temperature(logits, temperature):
    """Scale logits by 1/temperature before softmax sampling."""
    if temperature <= 0:
        raise ValueError("tempature must be positive")

    return logits / temperature

# Step 161 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Return logits with all but the top-k entries per row set to -inf."""
    vocab_size = logits.shape[-1]

    if k <= 0:
        raise ValueError("k must be positive")

    if k >= vocab_size:
        return logits.copy()

    # Indices of the k largest logits in each row.
    top_k_indices = np.argpartition(
        logits,
        -k,
        axis=-1,
    )[:, -k:]

    filtered = np.full(
        logits.shape,
        -np.inf,
        dtype=np.result_type(logits.dtype, float),
    )

    np.put_along_axis(
        filtered,
        top_k_indices,
        np.take_along_axis(logits, top_k_indices, axis=-1),
        axis=-1,
    )

    return filtered

# Step 162 - softmax_to_probs
def softmax_to_probs(logits):
    """Convert (1, V) logits into a row-wise probability distribution."""
    return stable_softmax_2d_rowwise(logits)

# Step 163 - sample_one_token
def sample_one_token(probs, rng):
    """Sample one token id from probs of shape (1, vocab_size) using rng."""
    vocab_size = probs.shape[1]

    token_id = rng.choice(
        vocab_size,
        p=probs[0],
    )

    return int(token_id)

# Step 164 - append_token_to_sequence
import numpy as np

def append_token_to_sequence(context_ids, token_id):
    """Append token_id as a new final column to context_ids of shape (1, T)."""
    new_token = np.array(
        [[token_id]],
        dtype=context_ids.dtype,
    )

    return np.concatenate(
        [context_ids, new_token],
        axis=1,
    )

# Step 165 - generation_loop_for_n_steps
def generation_loop_for_n_steps(
    params,
    prompt_ids,
    n_new_tokens,
    block_size,
    temperature,
    top_k,
    rng,
):
    """Autoregressively generate n_new_tokens from prompt_ids."""
    generated_ids = prompt_ids.copy()

    for _ in range(n_new_tokens):
        # Crop only the model input; preserve the full generated sequence.
        context_ids = crop_context_to_block_size(
            generated_ids,
            block_size,
        )

        logits = forward_to_get_logits(
            params,
            context_ids,
        )

        last_logits = take_last_position_logits(logits)
        scaled_logits = apply_temperature(last_logits, temperature)
        filtered_logits = top_k_filter(scaled_logits, top_k)
        probs = softmax_to_probs(filtered_logits)

        next_token_id = sample_one_token(probs, rng)

        generated_ids = append_token_to_sequence(
            generated_ids,
            next_token_id,
        )

    return generated_ids

# Step 166 - decode_final_sequence
def decode_final_sequence(generated_ids, itos):
    """Decode a (1, T) id tensor into a string using itos."""
    return decode_ids(generated_ids[0], itos)

