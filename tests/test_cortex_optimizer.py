"""Pin standard AdamW semantics for packed routed adapter parameters."""

import pytest
import torch

from .test_cortex_initialization_semantics import configured_branch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(previous_threads)


def routed_branch(kind):
    current, layer = configured_branch(kind, "routed", torch.float32)
    parameters = [p for name, p in current.named_parameters() if "cortex_" in name]
    with torch.no_grad():
        for parameter in parameters:
            parameter.normal_(std=0.2)
    assert all(parameter.requires_grad for parameter in parameters)
    return layer, parameters


def routed_loss(layer, active):
    inputs = torch.ones(4, 8)
    indices = torch.tensor([active] * 4)
    weights = torch.tensor([[0.7, 0.3]] * 4)
    return layer(inputs, indices, weights).square().mean()


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_packed_adamw_decays_unselected_slices_with_zero_gradients(kind):
    layer, parameters = routed_branch(kind)
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.2)
    before = [parameter[2].detach().clone() for parameter in parameters]
    routed_loss(layer, [0, 1]).backward()
    assert any(parameter.grad[:2].abs().sum() > 0 for parameter in parameters)
    for parameter in parameters:
        assert torch.count_nonzero(parameter.grad[2]) == 0
    optimizer.step()
    for parameter, original in zip(parameters, before):
        torch.testing.assert_close(parameter[2], original * 0.98, rtol=1e-6, atol=1e-7)
        assert optimizer.state[parameter]["step"] == 1


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_packed_adamw_delayed_expert_uses_shared_counter_and_inactive_momentum(kind):
    layer, parameters = routed_branch(kind)
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
    packed = layer.cortex_gate["default"].weight
    first_visit = torch.nn.Parameter(packed[2].detach().clone())
    independent = torch.optim.AdamW([first_visit], lr=0.01, weight_decay=0)
    routed_loss(layer, [0, 1]).backward()
    optimizer.step()
    torch.testing.assert_close(packed[2], first_visit, rtol=0, atol=0)
    assert optimizer.state[packed]["step"] == 1
    assert first_visit not in independent.state
    assert torch.count_nonzero(optimizer.state[packed]["exp_avg"][0]) > 0
    before_inactive = packed[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    routed_loss(layer, [1, 2]).backward()
    assert torch.count_nonzero(packed.grad[0]) == 0
    assert packed.grad[2].abs().sum() > 0
    first_visit.grad = packed.grad[2].detach().clone()
    optimizer.step()
    independent.step()
    assert optimizer.state[packed]["step"] == 2
    assert independent.state[first_visit]["step"] == 1
    # Matching first-visit weights and data gradients do not imply the same
    # AdamW update when one tensor already has a shared step counter.
    assert (packed[2] - first_visit).abs().max() > 1e-6
    # A zero data gradient also permits an update from existing momentum.
    assert (packed[0] - before_inactive).abs().max() > 1e-6
