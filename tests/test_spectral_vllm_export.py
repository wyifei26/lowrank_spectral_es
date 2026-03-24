import torch

from models.spectral_vllm import spectral_matrix_to_lora


def test_spectral_matrix_to_lora_reconstructs_delta_weight():
    torch.manual_seed(0)
    out_features = 7
    in_features = 5
    rank = 3
    u_basis = torch.randn(out_features, rank)
    vh_basis = torch.randn(rank, in_features)
    matrix = torch.randn(rank, rank)

    lora_a, lora_b = spectral_matrix_to_lora(
        u_basis=u_basis,
        vh_basis=vh_basis,
        matrix=matrix,
    )

    expected = u_basis @ matrix @ vh_basis
    actual = lora_b @ lora_a
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
