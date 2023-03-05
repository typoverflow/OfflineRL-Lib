# OfflineRL-Lib
> 🚧 This repo is not ready for release, benchmarking is ongoing. 🚧

OfflineRL-Lib provides unofficial and benchmarked PyTorch implementations for selected OfflineRL algorithms, including: 
- [Extreme Q-Learning](https://arxiv.org/abs/2301.02328)
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [AWAC](https://arxiv.org/abs/2006.09359)

still benchmarking ... 
- [EDAC](https://arxiv.org/abs/2110.01548)
- [SAC-N](https://arxiv.org/abs/2110.01548)
- [TD3-BC](https://arxiv.org/pdf/2106.06860.pdf)

under developing (model based algorithms) ...
- [MOPO](https://arxiv.org/abs/2005.13239)
- [MAPLE](https://proceedings.neurips.cc/paper/2021/file/470e7a4f017a5476afb7eeb3f8b96f9b-Paper.pdf)
- [RAMBO](https://arxiv.org/abs/2204.12581)


## Benchmark Results
See [reproduce/benchmark_result.md](https://github.com/typoverflow/blob/master/benchmark_result.md) for details. 

## Citing OfflineRL-Lib
If you use OfflineRL-Lib in your work, please use the following bibtex
```tex
@misc{offinerllib,
  author = {Chenxiao Gao},
  title = {OfflineRL-Lib: Benchmarked Implementations of Offline RL Algorithms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/typoverflow/OfflineRL-Lib}},
}
```

## Acknowledgements
We thank [CORL](https://github.com/tinkoff-ai/CORL) for providing finetuned hyper-parameters. 