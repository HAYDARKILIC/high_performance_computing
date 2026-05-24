# High-Performance Computing & Scaling Large Models

A graduate-level course on **High-Performance Computing (HPC)** and the systems engineering required to **train, fine-tune, and serve Large Language Models (LLMs)** efficiently. The course integrates GPU architecture, CUDA kernel programming, memory-efficient attention, distributed training paradigms, and modern inference systems into a unified, hands-on curriculum.

---

## Course Overview

The exponential growth of model parameters in modern deep learning has shifted the central bottleneck from *algorithmic novelty* to *systems efficiency*. Training a state-of-the-art language model is no longer a single-GPU task; it is a distributed-systems problem governed by the limits of memory bandwidth, interconnect latency, and arithmetic intensity. This course equips students with the theoretical foundations and practical engineering skills required to operate at this frontier.

Each week pairs a rigorous **lecture** on a core HPC topic with a **practical laboratory** implemented as a Jupyter notebook. Students will write low-level CUDA kernels, profile real models with NVIDIA Nsight, integrate production-grade inference engines such as vLLM, and scale training across multiple GPUs using ZeRO, FSDP, and 3D parallelism.

---

## Syllabus

| Week | Lecture Topic | Practical Laboratory |
|------|---------------|----------------------|
| 1 | HPC Foundations, Hardware Architectures, and Profiling | Model analysis with PyTorch Profiler and NVIDIA Nsight |
| 2 | Advanced PyTorch Optimization and Introduction to CUDA | Writing a matrix multiplication (`Y = X · W`) CUDA kernel from scratch |
| 3 | Memory-Bound Bottlenecks: FlashAttention and Serving Systems | vLLM (PagedAttention) integration and inference benchmarking |
| 4 | Memory Optimization in Distributed Training (ZeRO & FSDP) | DeepSpeed ZeRO-3 and dataset prefetching/caching |
| 5 | Multi-Dimensional Parallelism (3D) and Large-Cluster Scaling | Tensor and Pipeline parallelism with Megatron-LM |
| 6 | Efficient Inference, Model Compression, and Fine-Tuning Technologies | Distributed fine-tuning with LoRA/QLoRA and quantization |

---

## Repository Structure

```
high_performance_computing/
├── notebooks/                    # Weekly Jupyter notebooks (theory + practice)
│   ├── week1_hpc_foundations_profiling.ipynb
│   ├── week2_pytorch_optimization_cuda.ipynb
│   ├── week3_flashattention_vllm.ipynb
│   ├── week4_zero_fsdp_distributed.ipynb
│   ├── week5_3d_parallelism_megatron.ipynb
│   └── week6_lora_qlora_quantization.ipynb
├── src/                          # Reusable library code
│   ├── cuda/                     # Custom CUDA kernels (matmul, attention)
│   ├── distributed/              # Distributed training utilities
│   ├── inference/                # Inference and serving helpers
│   └── utils/                    # Profiling, benchmarking, logging
├── scripts/                      # Launcher scripts
│   └── slurm/                    # SLURM job submission templates
├── configs/                      # YAML configurations for training/inference
├── docker/                       # Reproducible Docker environment
├── tests/                        # Unit and integration tests
├── .github/workflows/            # Continuous Integration pipelines
├── assets/                       # Figures and diagrams
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment specification
├── pyproject.toml                # Project metadata and tooling
├── CONTRIBUTING.md               # Contribution guidelines
├── CODE_OF_CONDUCT.md
├── CITATION.cff                  # Academic citation metadata
├── LICENSE
└── README.md
```

---

## Learning Outcomes

By the end of this course, students will be able to:

1. **Analyze** the performance of a deep learning model using hardware-aware profiling tools (Nsight Systems, Nsight Compute, PyTorch Profiler) and identify whether a workload is compute-bound, memory-bound, or communication-bound.
2. **Implement** custom CUDA kernels for fundamental linear algebra operations and reason about occupancy, shared memory tiling, and warp-level primitives.
3. **Deploy** memory-efficient attention mechanisms (FlashAttention, PagedAttention) and explain their algorithmic and systems-level innovations.
4. **Design** distributed training strategies combining data parallelism, tensor parallelism, pipeline parallelism, and ZeRO sharding for models that exceed single-device memory.
5. **Apply** parameter-efficient fine-tuning (LoRA, QLoRA) and post-training quantization to deploy LLMs under realistic hardware budgets.

---

## Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with Compute Capability ≥ 7.0 (Volta or newer). Multi-GPU recommended for Weeks 4–6.
- **CUDA Toolkit**: 12.1 or newer.
- **Python**: 3.10 or newer.
- **Operating System**: Linux (Ubuntu 22.04 LTS verified). WSL2 supported with caveats.

### Installation

```bash
# Clone the repository
git clone https://github.com/HAYDARKILIC/high_performance_computing.git
cd high_performance_computing

# Option A: pip + virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate high_performance_computing

# Option C: Docker
docker compose -f docker/docker-compose.yml up -d
docker exec -it high_performance_computing bash
```

### Launching the Notebooks

```bash
jupyter lab notebooks/
```

For SLURM-managed clusters, see `scripts/slurm/` for ready-to-use submission templates.

---

## Recommended Hardware per Week

| Week | Minimum | Recommended |
|------|---------|-------------|
| 1 | 1× T4 (16 GB) | 1× A100 (40 GB) |
| 2 | 1× T4 | 1× A100 |
| 3 | 1× A10G (24 GB) | 1× A100 (80 GB) |
| 4 | 2× A100 (40 GB) | 4× A100 (80 GB) |
| 5 | 4× A100 (40 GB) | 8× A100 (80 GB) + NVLink |
| 6 | 1× A10G | 2× A100 |

---

## References and Further Reading

Key references that anchor the course material:

- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP.
- Rajbhandari, S. et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC20.
- Narayanan, D. et al. (2021). *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC21.
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.

---

## License

This course is released under the [MIT License](LICENSE). Course materials and lecture notes are additionally distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
