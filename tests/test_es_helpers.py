import torch

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
