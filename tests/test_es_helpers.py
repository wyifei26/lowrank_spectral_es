import torch

from es.cma import PerLayerCMAES
from es.noise import sample_antithetic_normal, sample_standard_normal
from es.spectral_update import (
    apply_alpha_update_to_direction_payloads,
    compute_gaussian_direction_payloads,
    compute_pairwise_direction_payloads,
    payload_global_norm,
)


def test_antithetic_noise_pairs():
    noise = sample_antithetic_normal(
        shape=(2, 3),
        num_mutants=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(noise[0], -noise[2])
    assert torch.allclose(noise[1], -noise[3])


def test_standard_normal_noise_shape():
    noise = sample_standard_normal(
        shape=(2, 3),
        num_mutants=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert noise.shape == (5, 2, 3)


def test_pairwise_direction_uses_antithetic_reward_differences():
    noise_payloads = {
        "layer": {
            "m": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
    }
    rewards = torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float32)

    direction_payloads, stats = compute_pairwise_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=rewards,
    )

    expected = torch.tensor([[0.125, -0.25], [0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(direction_payloads["layer"]["m"], expected)
    assert stats["pair_diff_mean"] == -0.25
    assert stats["pair_advantage_mean"] == -0.125
    assert stats["pair_nonzero_rate"] == 1.0


def test_apply_alpha_update_matches_alpha_over_sigma_scaling():
    direction_payloads = {
        "layer": {
            "m": torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32),
        }
    }

    step_payloads, stats = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_payloads,
        alpha_config={"m": 0.5},
        sigma_config={"m": 0.25},
    )

    expected = torch.tensor([[6.0, 8.0], [0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(step_payloads["layer"]["m"], expected)
    assert abs(payload_global_norm(step_payloads) - 10.0) < 1e-6
    assert abs(stats["raw_step_global_norm"] - 10.0) < 1e-6
    assert abs(stats["alpha_over_sigma"] - 2.0) < 1e-6
    assert stats["step_layers_clipped"] == 0.0


def test_apply_alpha_update_respects_layer_clip():
    direction_payloads = {
        "layer_a": {
            "m": torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32),
        },
        "layer_b": {
            "m": torch.tensor([[0.0, 0.0], [6.0, 8.0]], dtype=torch.float32),
        },
    }

    step_payloads, stats = apply_alpha_update_to_direction_payloads(
        direction_payloads=direction_payloads,
        alpha_config={"m": 0.5},
        sigma_config={"m": 0.1},
        max_layer_step_config={"m": 1.0},
    )

    assert torch.linalg.vector_norm(step_payloads["layer_a"]["m"]).item() <= 1.000001
    assert torch.linalg.vector_norm(step_payloads["layer_b"]["m"]).item() <= 1.000001
    assert stats["step_layer_max_norm"] <= 1.000001
    assert stats["step_layers_clipped"] == 2.0


def test_gaussian_direction_uses_mean_reward_weighted_noise():
    noise_payloads = {
        "layer": {
            "m": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                    [[1.0, 1.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
    }
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    direction_payloads, stats = compute_gaussian_direction_payloads(
        noise_payloads=noise_payloads,
        rewards=rewards,
    )
    expected = torch.tensor([[4.0 / 3.0, 5.0 / 3.0], [0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(direction_payloads["layer"]["m"], expected)
    assert stats["reward_mean"] == 2.0
    assert stats["reward_nonzero_rate"] == 1.0


def test_per_layer_cma_sampling_keeps_antithetic_pairs():
    cma = PerLayerCMAES(
        layer_shapes={"layer": torch.Size([2, 2])},
        sigma_config={"m": 0.5},
    )

    noise_payloads = cma.sample_noise(4, antithetic=True)
    transformed = noise_payloads["layer"]["m"]
    standard = noise_payloads["layer"]["m_z"]

    torch.testing.assert_close(standard[0], -standard[2])
    torch.testing.assert_close(standard[1], -standard[3])
    torch.testing.assert_close(transformed[0], -transformed[2])
    torch.testing.assert_close(transformed[1], -transformed[3])


def test_per_layer_cma_update_returns_clipped_step_and_updates_internal_state():
    cma = PerLayerCMAES(
        layer_shapes={"layer": torch.Size([2, 2])},
        sigma_config={"m": 0.5},
        cma_config={"selection_ratio": 0.5, "mean_step_scale": 1.0, "max_sigma": 10.0},
    )
    noise_payloads = {
        "layer": {
            "m": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "m_z": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        }
    }
    rewards = torch.tensor([4.0, 3.0, 1.0, 0.0], dtype=torch.float32)

    step_payloads, direction_stats, step_stats = cma.apply_update(
        rewards=rewards,
        noise_payloads=noise_payloads,
        current_states={"layer": torch.zeros(2, 2, dtype=torch.float32)},
        max_layer_step_config={"m": 0.3},
        max_state_norm=0.4,
    )

    weights = torch.tensor(
        [
            torch.log(torch.tensor(2.5)) - torch.log(torch.tensor(1.0)),
            torch.log(torch.tensor(2.5)) - torch.log(torch.tensor(2.0)),
        ],
        dtype=torch.float32,
    )
    weights /= weights.sum()
    expected_raw = 0.5 * torch.tensor([[weights[0], weights[1]], [0.0, 0.0]], dtype=torch.float32)
    expected_step = expected_raw * (0.3 / torch.linalg.vector_norm(expected_raw))

    torch.testing.assert_close(step_payloads["layer"]["m"], expected_step)
    assert direction_stats["cma_selected_reward_mean"] == 3.5
    assert step_stats["step_layers_clipped"] == 1.0
    assert step_stats["step_layer_max_norm"] <= 0.300001
    assert cma.layers["layer"].generation == 1
    assert float(cma.layers["layer"].sigma.item()) > 0.0


def test_per_layer_cma_state_dict_round_trip_preserves_layer_statistics():
    source = PerLayerCMAES(
        layer_shapes={"layer": torch.Size([2, 2])},
        sigma_config={"m": 0.25},
        cma_config={"selection_ratio": 0.5, "mean_step_scale": 0.8, "max_sigma": 5.0},
    )
    target = PerLayerCMAES(
        layer_shapes={"layer": torch.Size([2, 2])},
        sigma_config={"m": 0.25},
    )

    source.layers["layer"].sigma = torch.tensor(0.7, dtype=torch.float32)
    source.layers["layer"].cov = torch.tensor(
        [[2.0, 0.1, 0.0, 0.0], [0.1, 1.5, 0.0, 0.0], [0.0, 0.0, 1.2, 0.2], [0.0, 0.0, 0.2, 0.9]],
        dtype=torch.float32,
    )
    source.layers["layer"].chol = torch.linalg.cholesky(source.layers["layer"].cov)
    source.layers["layer"].p_sigma = torch.tensor([0.2, -0.1, 0.0, 0.3], dtype=torch.float32)
    source.layers["layer"].p_c = torch.tensor([0.0, 0.1, 0.2, -0.2], dtype=torch.float32)
    source.layers["layer"].generation = 4

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.layers["layer"].sigma, source.layers["layer"].sigma)
    torch.testing.assert_close(target.layers["layer"].cov, source.layers["layer"].cov)
    torch.testing.assert_close(target.layers["layer"].chol, source.layers["layer"].chol)
    torch.testing.assert_close(target.layers["layer"].p_sigma, source.layers["layer"].p_sigma)
    torch.testing.assert_close(target.layers["layer"].p_c, source.layers["layer"].p_c)
    assert target.layers["layer"].generation == 4
