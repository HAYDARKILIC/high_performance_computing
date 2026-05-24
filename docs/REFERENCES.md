# References

A consolidated bibliography of works cited across the six course modules.

## Hardware & Profiling (Week 1)

- Williams, S., Waterman, A., and Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4), 65–76.
- NVIDIA Corporation. (2022). *NVIDIA A100 Tensor Core GPU Architecture Whitepaper*.
- NVIDIA Corporation. (2023). *NVIDIA H100 Tensor Core GPU Architecture Whitepaper*.
- NVIDIA Corporation. *NVIDIA Nsight Systems User Guide*. https://docs.nvidia.com/nsight-systems/

## PyTorch Optimization & CUDA (Week 2)

- Harris, M. (2013). "An Even Easier Introduction to CUDA." NVIDIA Developer Blog.
- NVIDIA Corporation. *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Micikevicius, P. et al. (2018). "Mixed Precision Training." *ICLR 2018*.
- Tillet, P., Kung, H. T., and Cox, D. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." *MAPL 2019*.
- Ansel, J. et al. (2024). "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." *ASPLOS 2024*.

## FlashAttention & Serving (Week 3)

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
- Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*.
- Shah, J. et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." *arXiv:2407.08608*.
- Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*.
- Yu, G.-I. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.

## Distributed Training: Memory (Week 4)

- Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC 2020*.
- Rajbhandari, S. et al. (2021). "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning." *SC 2021*.
- Zhao, Y. et al. (2023). "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." *VLDB 2023*.
- Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016). "Training Deep Nets with Sublinear Memory Cost." *arXiv:1604.06174*.

## Multi-Dimensional Parallelism (Week 5)

- Shoeybi, M. et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv:1909.08053*.
- Narayanan, D. et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." *SC 2021*.
- Huang, Y. et al. (2019). "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism." *NeurIPS 2019*.
- Harlap, A. et al. (2018). "PipeDream: Fast and Efficient Pipeline Parallel DNN Training." *arXiv:1806.03377*.
- Korthikanti, V. et al. (2023). "Reducing Activation Recomputation in Large Transformer Models." *MLSys 2023*.

## Efficient Inference & Fine-Tuning (Week 6)

- Hu, E. J. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
- Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
- Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
- Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
- Lin, J. et al. (2024). "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*.
- Xiao, G. et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*.

## Frameworks & Tools

- Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS 2019*.
- Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. (2020). "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters." *KDD 2020*.
- Mangrulkar, S. et al. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods*. https://github.com/huggingface/peft
- Wolf, T. et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." *EMNLP 2020*.
