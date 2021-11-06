import numpy as np
import sonnet as snt
import tensorflow as tf

import transformers

def assert_tensors_close(t1, t2, **kwargs):
  np.testing.assert_allclose(t1.numpy(), t2.numpy(), **kwargs)

def test_head_to_batch_roundtrip(H=2, O=6, B=5, S=7):
  mha_block = transformers.MultiHeadAttentionBlock(
      num_heads=H,
      output_size=O,
      mem_size=0,
  )

  x = tf.random.uniform([B, S, O])
  to_batch = mha_block._heads_to_batch(x)
  to_output = mha_block._heads_to_output(to_batch)
  assert_tensors_close(x, to_output)

def test_mha_unroll_vs_step(H=2, O=6, B=1, S=5, M=0):
  mha_block = transformers.MultiHeadAttentionBlock(
      num_heads=H,
      output_size=O,
      mem_size=M,
  )

  inputs = tf.random.uniform([B, S, O])
  initial_state = mha_block.initial_state(B)

  bulk_outputs, bulk_final_state = mha_block(inputs, initial_state)

  step_outputs = []
  hidden_state = initial_state

  for i, step_input in enumerate(tf.unstack(inputs, axis=1)):
    step_input = tf.expand_dims(step_input, 1)
    outputs, hidden_state = mha_block(step_input, hidden_state)
    assert_tensors_close(outputs, bulk_outputs[:, i:i+1])
  
  tf.nest.map_structure(assert_tensors_close, hidden_state, bulk_final_state)

if __name__ == '__main__':
  test_head_to_batch_roundtrip()
  test_mha_unroll_vs_step()
