# This is a failed repo

# dLLM: Asynchronous Pipeline-Parallel LLM Orchestration Framework

An experimental, low-level LLM inference orchestration framework designed to implement asynchronous pipeline parallelism, block-wise INT8 quantization, and entropy-driven speculative branching across decentralized nodes. 

## architecture

this repository remains public as an engineering post-mortem detailing the micro-architectural and network-level bottlenecks that occur when attempting dynamic, dynamic-routed tensor parallelism over virtualized network infrastructure.

the framework attempted to distribute execution by routing intermediate activation tensors and dynamic Attention masks across distinct physical nodes during the layer-by-layer transformer forward pass.
- for an execution hidden dimension ($d$) of 4096 using 16-bit floating-point tensors, a single token's hidden state payload is highly compact (~16 KB). However, when evaluating the full history ($N$) during deep reasoning sequences, the **KV-Cache** scales linearly. Pushing multi-megabyte/gigabyte attention matrices over standard TCP/IP sockets saturates network interface card (NIC) egress channels instantly.
- GPU matrix multiplications (GEMM) occur on the scale of microseconds. Standard virtual networking introduces latency on the scale of milliseconds. The system inverted the compute-to-I/O ratio, causing the GPUs to spend **>95% of their clock cycles stalled** in idle execution bubbles waiting for network packets.

### entropy driven branching

to maximize utilization, the framework used an entropy-driven router to predict difficulty and speculatively branch logical execution pathways across the cluster.
- speculative execution requires a locked, predictable pipeline schedule to successfully mask data-transfer latency behind active compute streams. introducing dynamic branching based on runtime entropy calculation meant nodes could not pre-allocate VRAM or pre-fetch data buffers.
- the system introduced constant pipeline stalls and execution rollbacks. the synchronization overhead required to continuously validate speculative branches across nodes over a slow wire completely wiped out any performance gains achieved by distributing the workload.

## takeaways

1. pushing raw transformer intermediate states across physical network cables without dedicated data-center fabrics (e.g., 400Gbps InfiniBand, RoCEv2, NVLink) is a fundamental systems anti-pattern. 
2. to run distributed consumer inference, optimization routines must be strictly restricted to the **inter-prompt idle space**. Live, token-level network dependency during the forward pass turns a parallel computing cluster back into an incredibly slow serial wire.
3. inference optimization on constrained hardware is entirely a memory bandwidth and context-caching management problem, not a macro-routing problem.

