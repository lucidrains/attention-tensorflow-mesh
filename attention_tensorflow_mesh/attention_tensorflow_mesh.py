import math
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

# helpers

def default(val, d):
    return val if val is not None else d

# simple linear layer

def linear(x, dim_out, scope = 'linear', bias = True):
    with tf.variable_scope(scope):
        *_, dim_in = x.shape
        w_init_stdev = 1 / math.sqrt(dim_in.size)

        return  mtf.layers.dense(x, new_dims=[dim_out], reduced_dims=[dim_in], name=scope, use_bias=bias,
                                 kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=tf.float32))

# norm

def norm(x, axis = None, epsilon=1e-5):
    axis = default(axis, x.shape[-1])

    u = mtf.reduce_mean(x, reduced_dim=axis)
    s = mtf.reduce_mean(mtf.square(x - u))

    u = mtf.broadcast(u, x.shape)
    s = mtf.broadcast(s, x.shape)

    return (x - u) * mtf.rsqrt(s + epsilon)

def prenorm(fn):
    def inner(x, *args, **kwargs):
        return fn(norm(x), *args, **kwargs)
    return inner

def residual(fn):
    def inner(x, *args, **kwargs):
        return fn(x, *args, **kwargs) + x
    return inner

# full multi-head attention

def attention(x, dim_head, dim_features_head, scope = 'attn', causal = False):
    with tf.variable_scope(scope):
        mesh, batch, seq, dim = x.mesh, *x.shape

        dim_heads = mtf.Dimension('dim_heads', dim_head.size * dim_features_head.size)
        dim_intermediate = mtf.Dimension('qkv_dimension', dim_heads.size * 3)
        qkv = linear(x, dim_intermediate, bias = False, scope='to_qkv')

        q, k, v = mtf.split(qkv, dim_intermediate, 3)
        q, k, v = map(lambda t: mtf.reshape(t, [batch, seq, dim_head, dim_features_head]), (q, k, v))
        q, k, v = map(lambda t: mtf.transpose(t, [batch, dim_head, seq, dim_features_head]), (q, k, v))

        k, v = map(lambda t: mtf.rename_dimension(t, seq.name, 'memory_length'), (k, v))
        mem_len_dim = v.shape[-2]

        dots = mtf.layers.us_einsum([q, k], [batch, dim_head, seq, mem_len_dim])

        if causal:
            i = mtf.range(mesh, seq, tf.int32)
            j = mtf.range(mesh, mem_len_dim, tf.int32)
            i, j = map(lambda t: mtf.broadcast(t, [seq, mem_len_dim]), (i, j))
            mask = mtf.less(i, j + mem_len_dim.size - seq.size)
            mask = mtf.cast(mask, tf.float32) * -1e9
            dots += mask

        attn = mtf.softmax(dots, mem_len_dim)
        out = mtf.einsum([dots, v], [batch, dim_head, seq, dim_features_head])

        out = mtf.transpose(out, [batch, seq, dim_head, dim_features_head])
        out = mtf.reshape(out, [batch, seq, dim_heads])

        combined_out = linear(out, dim, scope='combine_output')
        return combined_out

# feed forward

def ff(x, mult = 4, scope = 'ff'):
    *_, dim = x.shape

    with tf.variable_scope(scope):
        dim_intermediate = mtf.Dimension('ff_intermediate', dim.size * mult)
        h = linear(x, dim_intermediate, scope='w1')
        h = mtf.gelu(h)
        h = linear(h, dim, scope='w2')
        return h

# block

def transformer(x, *, depth, dim_head, dim_features_head, causal = False):
    attn_fn = residual(prenorm(attention))
    ff_fn = residual(prenorm(ff))

    for i in range(depth):
        with tf.variable_scope(f'layer_{i}'):
            x = attn_fn(x, dim_head, dim_features_head, causal = causal)
            x = ff(x)
    return x
