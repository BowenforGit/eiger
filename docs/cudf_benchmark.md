# Run cuDF microbenchmarks

To ease the benchmark develop without introducing too much overhead, we use the
pylibcudf to build the microbenchmarks. 

## Requirement

We need to install cuDF in python. Follow the instruction in [Github](https://github.com/rapidsai/cudf).

Here, we use the `conda` package manager.

```bash
conda create --name cudf_bench
conda activate cudf_bench

# if you use CUDA 12
conda install -c rapidsai cudf

# if you use CUDA 13
pip install cudf-cu13
```

