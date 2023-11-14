Comparing GPU performance of jax and torch.compile on some standard ML models/layers. Mostly for fun.

The benchmark does a full forward + backward pass, and enables tensorfloat32 matmuls for both frameworks.
There is a full sync after each forward+backward.

Results on my setup (YMMV, the many benchmarking caveats apply):

```
attn_seq1024_dim512_f16
-----------------------
pytorch:  9.5ms ± 0.20ms p90=9.7ms
jax:     11.5ms ± 0.72ms p90=12.1ms

attn_seq1024_dim512_tf32
------------------------
pytorch: 19.6ms ± 1.60ms p90=20.6ms
jax:     23.0ms ± 2.20ms p90=26.5ms

attn_seq2048_dim256_tf32
------------------------
pytorch: 23.6ms ± 0.99ms p90=23.8ms
jax:     25.7ms ± 2.22ms p90=29.7ms

resnet50
--------
pytorch: 58.5ms ± 2.89ms p90=61.6ms
jax:     62.2ms ± 6.87ms p90=64.1ms
```

Note: attention impl is from scratch (both in a similar manner), not using any built-in modules, to make this a test of the compiler and not the library.
