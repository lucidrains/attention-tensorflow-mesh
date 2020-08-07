import math
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

# simple linear layer

def linear(x, dim_out, scope = 'linear', bias = True):
    *_, dim_in = x.shape
    w_init_stdev = 1 / math.sqrt(dim_in.size)

    return  mtf.layers.dense(x, new_dims=[dim_out], reduced_dims=[dim_in], name=scope, use_bias=bias,
                             kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=tf.float32))

# full non-causal attention

def attention(x, dim_head, dim_features_head, scope = 'attn'):
    batch, seq, dim = x.shape

    dim_heads = mtf.Dimension('dim_heads', dim_head.size * dim_features_head.size)
    dim_intermediate = mtf.Dimension('qkv_dimension', dim_heads.size * 3)
    qkv = linear(x, dim_intermediate, bias = False, scope='to_qkv')

    q, k, v = mtf.split(qkv, dim_intermediate, 3)
    q, k, v = map(lambda t: mtf.reshape(t, [batch, seq, dim_head, dim_features_head]), (q, k, v))
    q, k, v = map(lambda t: mtf.transpose(t, [batch, dim_head, seq, dim_features_head]), (q, k, v))

    k, v = map(lambda t: mtf.rename_dimension(t, seq.name, 'memory_length'), (k, v))
    mem_len_dim = v.shape[-2]

    dots = mtf.layers.us_einsum([q, k], [batch, dim_head, seq, mem_len_dim])
    attn = mtf.softmax(dots, mem_len_dim)
    out = mtf.einsum([dots, v], [batch, dim_head, seq, dim_features_head])

    out = mtf.transpose(out, [batch, seq, dim_head, dim_features_head])
    out = mtf.reshape(out, [batch, seq, dim_heads])

    combined_out = linear(out, dim, scope='combine_output')
    return combined_out
