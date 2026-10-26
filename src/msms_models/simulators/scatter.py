"""The torch_scatter reductions ICEBERG and GLACIER use, on native PyTorch scatter operations.

torch-scatter's wheels for PyTorch 2.7 need glibc 2.32, which RHEL 8 and Ubuntu 20.04
clusters lack. These give the same results as torch-scatter 2.1.2: sums use the same
``scatter_add_`` kernel, max/min fill empty slots with 0, and the softmax follows
torch-scatter's recipe (including its 1e-12 epsilon).
"""



def _output(src, index, dim, dim_size):
    dim = dim % src.dim()
    size = list(src.shape)
    size[dim] = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    return src.new_zeros(size), dim


def scatter_add(src, index, dim=-1, dim_size=None):
    out, dim = _output(src, index, dim, dim_size)
    return out.scatter_add_(dim, index, src)


def scatter(src, index, dim=-1, dim_size=None, reduce="sum"):
    assert reduce == "sum"
    return scatter_add(src, index, dim, dim_size)


def _scatter_extreme(src, index, dim, dim_size, reduce):
    out, dim = _output(src, index, dim, dim_size)
    # Only the values are used; the arg indices torch-scatter also returns are not.
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False), None


def scatter_max(src, index, dim=-1, dim_size=None):
    return _scatter_extreme(src, index, dim, dim_size, "amax")


def scatter_min(src, index, dim=-1, dim_size=None):
    return _scatter_extreme(src, index, dim, dim_size, "amin")


def scatter_softmax(src, index, dim=-1, eps=1e-12):
    maximum = scatter_max(src, index, dim)[0].gather(dim, index)
    scores = (src - maximum).exp_()
    total = scatter_add(scores, index, dim).add_(eps).gather(dim, index)
    return scores.div(total)
