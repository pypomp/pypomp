Fast Random Number Generation on GPUs
========================

The simulators included in the JAX package often suffer from warp divergence due to using rejection sampling. For the user's convenience, we have included a few replacement functions that use Inverse Transform Sampling to generate random variables. These functions use JAX under the hood, so they can be used in a JIT-compiled context. While the following functions include some branching in order to handle edge cases, the performance loss from warp divergence is minimal. 

What is warp divergence?
-------------------------

Warp divergence is a performance-degrading phenomenon on GPUs that occurs when threads within the same warp (typically a group of 32 threads) encounter a conditional branch, such as an if/else statement, and follow different execution paths. Because GPU hardware uses a Single Instruction, Multiple Thread (SIMT) model, all threads in a warp share a single program counter; if they diverge, the GPU must serialize the paths, executing each one sequentially while masking out the inactive threads, which can lead to significant drops in throughput. 

Why is warp divergence relevant here?
--------------------------------------

In the context of distribution simulators (random number samplers), this occurs most frequently in rejection sampling algorithms (e.g., Marsaglia and Tsang's Gamma sampler). In these simulators, if a single thread in a warp has its proposed value rejected, the entire warp is forced to repeat the loop iteration, even if the other 31 threads have already successfully sampled their values. To avoid this, high-performance simulators often employ Inverse Transform Sampling, which uses a deterministic, branch-free mathematical transformation of a uniform random variable to the target distribution, ensuring all threads in the warp stay in lock-step.



.. automodule:: pypomp.random
   :members:
   :undoc-members:
   :show-inheritance:

