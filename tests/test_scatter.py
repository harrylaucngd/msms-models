import pytest

torch = pytest.importorskip("torch")

from msms_models.simulators import scatter  # noqa: E402


def test_reductions_match_per_group_computation():
    src = torch.tensor([[1.0, -2.0, 3.0, 0.5, 4.0]])
    index = torch.tensor([[0, 2, 0, 2, 3]])
    total = scatter.scatter_add(src, index, dim=-1, dim_size=5)
    assert total.tolist() == [[4.0, 0.0, -1.5, 4.0, 0.0]]
    # Empty slots are 0, as in torch-scatter.
    assert scatter.scatter_max(src, index, dim=-1, dim_size=5)[0].tolist() == [[3.0, 0.0, 0.5, 4.0, 0.0]]
    assert scatter.scatter_min(src, index, dim=-1, dim_size=5)[0].tolist() == [[1.0, 0.0, -2.0, 4.0, 0.0]]
    softmax = scatter.scatter_softmax(src, index, dim=-1)
    for group in (0, 2, 3):
        members = index[0] == group
        expected = torch.softmax(src[0, members], dim=0)
        assert torch.allclose(softmax[0, members], expected)
