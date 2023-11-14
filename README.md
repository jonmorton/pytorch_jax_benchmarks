Comparing GPU performance of jax and torch.compile on some standard ML models/layers. Mostly for fun.

The benchmark does a full forward + backward pass, and enables tensorfloat32 matmuls for both frameworks.
There is a full sync after each forward+backward.

Results on my setup (YMMV, the many benchmarking caveats apply):

```
attn_seq1024_dim512_f16
-----------------------
pytorch:  5.5ms ± 0.61ms p90=6.6ms
jax:      6.1ms ± 0.44ms p90=6.6ms

attn_seq1024_dim512_tf32
------------------------
pytorch: 10.0ms ± 0.34ms p90=10.3ms
jax:     12.4ms ± 1.34ms p90=14.5ms

attn_seq2048_dim256_tf32
------------------------
pytorch: 13.8ms ± 1.93ms p90=16.9ms
jax:     14.4ms ± 1.62ms p90=16.9ms

resnet50
--------
pytorch: 32.9ms ± 0.30ms p90=33.1ms
jax:     38.2ms ± 8.42ms p90=43.6ms
```

Note: attention impl is from scratch (both in a similar manner), not using any built-in modules, to make this a test of the compiler and not the library.
