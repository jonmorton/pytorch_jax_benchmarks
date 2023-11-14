Comparing GPU performance of jax and torch.compile on some standard ML models/layers. Mostly for fun.

The benchmark does a full forward + backward pass, and enables tensorfloat32 matmuls for both frameworks.
Each forward+backward is fully synchronized.

Results on my setup (YMMV):

```
  attn_seq1024_dim512
    pytorch: 11.5ms±0.79ms
    jax: 13.8ms±2.06ms
  attn_seq2048_dim256
    pytorch: 14.7ms±1.02ms
    jax: 15.7ms±2.16ms
  resnet50
    pytorch: 37.6ms±2.96ms
    jax: 40.8ms±8.66ms
```

Note: attention impl is from scratch, not using any built-in modules, to make this a test of the compiler and not the library.
