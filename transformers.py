import sonnet as snt
import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

def positional_encoding(seq_len, d_model, batch_size=1):
    """
    Returns a tensor following the postional encoding function
     (sinusodal from Vaswani et. all 2017).

    Return shape: [batch, seq_len, d_model]
    """
    def encoding_angle(pos, i):
        pos = tf.cast(pos, tf.dtypes.float32)
        i = tf.cast(i, tf.dtypes.float32)
        d = tf.cast(d_model, tf.dtypes.float32)
        denom = tf.math.pow(10000., 2. * i/d)
        return pos / denom

    i_tensor = tf.expand_dims(tf.range(0, d_model//2), 0) # [1, d_model/2]
    i_tensor = tf.broadcast_to(i_tensor, [seq_len, d_model//2]) # [seq_len, d_model/2]
    j_tensor = tf.expand_dims(tf.range(0, seq_len), 1) # [seq_len, 1]
    j_tensor = tf.broadcast_to(j_tensor, [seq_len, d_model//2]) # [seq_len, d_model/2]
    angles = encoding_angle(j_tensor, i_tensor) # [seq_len, d_model/2]

    # Apply sin to even indices, cos to odd indices
    sins_angle = tf.math.sin(angles) # [seq_len, d_model/2]
    coss_angles = tf.math.cos(angles) # [seq_len, d_model/2]
    joined = tf.concat([sins_angle, coss_angles], -1) # [s, d]

    #Add in batch
    encoding = tf.expand_dims(joined, 0)
    encoding = tf.repeat(encoding, [batch_size], axis=0) # [b, s, d]
    return encoding

def attention(queries, keys, values, mem_size: int):
    """
    Returns the 'attention' between three sequences: keys, queries, and values
    Specifically this implementation uses 'scaled dot-product' attention.

    This can be seen as a measure of the compatibility or relative importance between the keys and queries.
    This compatilbility is then applied to the 'input' sequence represented by values.

    Returns a tensor with the same shape as Values where [b, i ,j] represents the relative importance
    "attention" of element j in the sequence.

    keys: (batch, mem_size + seq_len, D_k)
    queries: (batch, seq_len, D_k)
    values: (batch, mem_size + seq_len, D_v)
    returns: (batch, seq_len, D_v)
    """
    tf.debugging.assert_shapes([
        (keys, ('B', 'MpS', 'D_k')),
        (queries, ('B', 'S', 'D_k')),
        (values, ('B', 'MpS', 'D_v')),
    ])
    B, MpS, D_k = tf.unstack(tf.shape(keys))
    B, S, D_k = tf.unstack(tf.shape(queries))

    # compat [b, i, j] is the dot product of key i and query j (for batch # b)
    compat = tf.matmul(queries, keys, transpose_b=True) # [B, S, M+S]
    norm_compat = compat / tf.sqrt(tf.cast(D_k, compat.dtype)) # [B, S, M+S]

    i = tf.expand_dims(tf.range(S), 1)  # [S, 1]
    j = tf.expand_dims(tf.range(MpS) - mem_size, 0)  # [1, M+S]
    # mask[i, j] == j \in [(i - mem_size), i]
    mask = tf.logical_and(i - mem_size <= j, j <= i)

    norm_compat = tf.where(mask, norm_compat, np.NINF)
    probs = tf.nn.softmax(norm_compat) # [B, S, M+S]
    att = tf.matmul(probs, values) # [B, S, D_v]
    return att

class MultiHeadAttentionBlock(snt.Module):
  def __init__(self, num_heads, output_size, mem_size, name='MultiHeadAttentionBlock'):
    super(MultiHeadAttentionBlock, self).__init__(name=name)
    self.num_heads = num_heads
    self.output_size = output_size
    assert output_size % num_heads == 0, "output_size must be a multiple of num_heads"
    self.projection_size = output_size // num_heads
    self.W_qkv = snt.Linear(3 * output_size)
    self.mem_size = mem_size

  def initial_state(self, batch_size: int):
    return dict(
        keys=tf.zeros([batch_size, self.mem_size, self.output_size]),
        values=tf.zeros([batch_size, self.mem_size, self.output_size]),
    )

  def _heads_to_batch(self, x):
    """
    Merges head dim into batch dim.

    x: [B, S, H * P]
    returns: [B * H, S, P]
    """
    B, S, O = tf.unstack(tf.shape(x))
    H = self.num_heads

    x = tf.reshape(x, [B, S, H, -1])  # unmerge heads from output
    x = tf.transpose(x, [0, 2, 1, 3])  # [B, H, S, P]
    x = tf.reshape(x, [B * H, S, -1])  # merge heads into batch
    return x

  def _heads_to_output(self, x):
    """
    Undoes _heads_to_batch.

    x: [B * H, S, P]
    returns: [B, S, H * P]
    """
    BH, S, P = tf.unstack(tf.shape(x))
    H = self.num_heads

    x = tf.reshape(x, [-1, H, S, P])  # unmerge heads from batch
    x = tf.transpose(x, [0, 2, 1, 3])  # [B, H, S, P]
    x = tf.reshape(x, [-1, S, H * P])  # merge heads into output
    return x


  def __call__(self, inputs, prev_state: dict):
    """
    For each head, this block will project input into 3 spaces (keys, queries, values)
    and subsequently run an attention block on each projection. The results of each heads are
    combined (via concat) into the final output.

    inputs: [B, S, D_m]
    prev_state: {keys: [B, M, D_m], values: [B, M, D_m]}
    Returns: (outputs: [B, S, D_m], next_state: shape(prev_state)}
    """
    B, S, D_m = tf.unstack(tf.shape(inputs))
    M = self.mem_size

    qkv = self.W_qkv(inputs)  # [B, S, 3 * O]
    q, k, v = tf.split(qkv, 3, axis=2)  # [B, S, O]

    prev_k = prev_state["keys"]  # [B, M, O]
    all_k = tf.concat([prev_k, k], 1)  # [B, M + S, O]

    prev_v = prev_state["values"]  # [B, M, O]
    all_v = tf.concat([prev_v, v], 1)  # [B, M + S, O]

    attention_kwargs = dict(
        keys=all_k,
        queries=q,
        values=all_v,
    )
    attention_kwargs = tf.nest.map_structure(
        self._heads_to_batch, attention_kwargs)

    o = attention(mem_size=M, **attention_kwargs)  # [B * H, S, P]
    o = self._heads_to_output(o)  # [B, S, O]

    next_state = dict(
        keys=all_k[:, S:],
        values=all_v[:, S:],
    )

    return o, next_state

class TransformerEncoderBlock(snt.Module):
  def __init__(
      self, output_size, ffw_size, num_heads, mem_size,
      name="EncoderTransformerBlock"):
    super(TransformerEncoderBlock, self).__init__(name=name)
    self.output_size = output_size
    self.ffw_size = ffw_size
    self.attention = MultiHeadAttentionBlock(num_heads, output_size, mem_size)
    self.feed_forward_in = snt.Linear(ffw_size)
    self.feed_forward_out = snt.Linear(output_size)
    self.norm_1 = snt.LayerNorm(-1, False, False)
    self.norm_2 = snt.LayerNorm(-1, False, False)

  def initial_state(self, batch_size):
    return self.attention.initial_state(batch_size)

  def __call__(self, inputs, prev_state):
    # MHAB
    att, next_state = self.attention(inputs, prev_state)
    # return att, next_state
    # Add (res) + LayerNorm
    res_norm_att = self.norm_1(att + inputs)
    # Feed forward
    feed_in = self.feed_forward_in(res_norm_att)
    act = tf.nn.relu(feed_in)
    feed_out = self.feed_forward_out(act)
    # Add (res) + LayerNorm
    output = self.norm_2(res_norm_att + feed_out)
    return output, next_state

class EncoderOnlyTransformer(snt.Module):
  def __init__(
      self,
      output_size: int,
      num_layers: int,
      ffw_size: int,
      num_heads: int,
      mem_size: int,
      name="EncoderTransformer"):
    super(EncoderOnlyTransformer, self).__init__(name=name)
    self.num_layers = num_layers
    self.transformer_blocks = []
    for _ in range(num_layers):
      t = TransformerEncoderBlock(output_size, ffw_size, num_heads, mem_size)
      self.transformer_blocks.append(t)
    # maybe add assertion about attention size and output_size
    self.shape_convert = snt.Linear(output_size)

  def initial_state(self, batch_size):
    return [t.initial_state(batch_size) for t in self.transformer_blocks]

  def __call__(self, inputs, prev_state):
    """
    inputs: [S, B, D]
    prev_state: L * [dict[B, M, D_m]]

    returns: (outputs: [S, B, D_m], next_state: shape(prev_state))
    """
    inputs = self.shape_convert(inputs)
    inputs = tf.transpose(inputs, [1, 0, 2])
    # i_shape = tf.shape(inputs)
    # encoding = positional_encoding(i_shape[1], i_shape[2], batch_size=i_shape[0])
    # x = inputs + encoding
    x = inputs
    next_state = []
    for t, p in zip(self.transformer_blocks, prev_state):
      x, n = t(x, p)
      next_state.append(n)
    x = tf.transpose(x, [1, 0, 2])
    return x, next_state
