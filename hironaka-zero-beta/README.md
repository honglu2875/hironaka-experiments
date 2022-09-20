AlphaZero and MuZero style approach is implemented in `hironaka.jax`. Two batches and a total of 100 quick runs (3000 gradient steps) were finished on the two simple network architectures:
- [256, 256, 256, 256]
- [1024, 1024, 1024, 1024]
Both are fully connected Dense network. Each run utilize 8 A100 in parallel (thus the real batch size and sample size are multiplied by 8).

The result is too huge and I cannot upload here, but the learning rate and batch size search suggests the following:
- for small network, learning rate ~ 1e-3, 1e-4 brings the best start (with confidence). Best batch size was 128 but variance was too high and could not tell. 
- for large network, learning rate ~ 1e-4, 1e-5 have the best start (with confidence). Batch size has too much variance, but 512 and 1024's performance was great.

