from typing import Callable

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def pairwise_indexed(feats: np.ndarray, pairs: np.ndarray,
                     batch_pairwise_op: Callable[[tf.Tensor, tf.Tensor],
                                                 tf.Tensor], batch_size):
    '''Compute pairwise operation'''
    featstf = tf.constant(feats)

    def get_pair(ids):
        feat_x = tf.gather(featstf, ids[:, 0])
        feat_y = tf.gather(featstf, ids[:, 1])
        return feat_x, feat_y

    dataset = tf.data.Dataset.from_tensor_slices(pairs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(get_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    result = []
    for x_sample, y_sample in tqdm(dataset, leave=False):
        output = batch_pairwise_op(tf.constant(x_sample),
                                   tf.constant(y_sample))
        result.append(output.numpy())

    result = np.concatenate(result)
    assert len(result) == len(pairs)
    return result


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
))
def pairwise_corr(x_sample: tf.Tensor, y_sample: tf.Tensor) -> tf.Tensor:
    import tensorflow_probability as tfp
    return tfp.stats.correlation(x_sample,
                                 y_sample,
                                 sample_axis=1,
                                 event_axis=None)


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
))
def pairwise_cosine(x_sample: tf.Tensor, y_sample: tf.Tensor) -> tf.Tensor:
    x_sample = tf.linalg.l2_normalize(x_sample, axis=1)
    y_sample = tf.linalg.l2_normalize(y_sample, axis=1)
    c_dist = tf.reduce_sum(x_sample * y_sample, axis=1)
    return c_dist


@tf.function
@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
))
def tf_random_binary_matrix(m: int, n: int, k: int):
    ''' Generate k random indices for each row'''
    # Initialize the matrix.
    ones = tf.ones((m, k), dtype=tf.float32)
    zeros = tf.zeros((m, n - k), dtype=tf.float32)
    matrix = tf.concat([ones, zeros], axis=1)
    matrix = tf.map_fn(tf.random.shuffle, matrix)
    return matrix


@tf.function
def tf_compute_ap(rel_k):
    '''Compute average precision based on binary list sorted by relevance'''
    tp = tf.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = tf.range(1, rel_k.shape[1] + 1, dtype=tf.float32)
    pr_k = tp / k
    ap = tf.reduce_sum(pr_k * rel_k, axis=1) / num_pos
    return ap


def random_binary_matrix(m: int, n: int, k: int):
    return tf_random_binary_matrix(tf.constant(m), tf.constant(n),
                                   tf.constant(k))


def compute_ap(rel_k):
    return tf_compute_ap(tf.constant(rel_k))
