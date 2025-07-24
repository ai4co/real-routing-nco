#!/usr/bin/env python3
"""
Test script to verify GatingNeuralAdaptiveBias and NaiveNeuralAdaptiveBias implementations
"""

import torch
from rrnco.models.nn.attn_freenet import GatingNeuralAdaptiveBias, NaiveNeuralAdaptiveBias, AttnFreeNet

def test_neural_adaptive_bias_types():
    """Test both GatingNeuralAdaptiveBias and NaiveNeuralAdaptiveBias implementations"""
    
    # Test parameters
    batch_size = 2
    num_nodes = 10
    embed_dim = 128
    
    # Create test data
    coords = torch.randn(batch_size, num_nodes, 2)
    cost_mat = torch.randn(batch_size, num_nodes, num_nodes)
    duration_mat = torch.randn(batch_size, num_nodes, num_nodes)
    
    print("Testing GatingNeuralAdaptiveBias and NaiveNeuralAdaptiveBias...")
    
    # Test GatingNeuralAdaptiveBias
    print("\n1. Testing GatingNeuralAdaptiveBias:")
    gating_nab = GatingNeuralAdaptiveBias(embed_dim=embed_dim, use_duration_matrix=True)
    result_gating = gating_nab(coords, cost_mat, duration_mat)
    print(f"   Output shape: {result_gating.shape}")
    print(f"   Output range: [{result_gating.min():.3f}, {result_gating.max():.3f}]")
    
    # Test NaiveNeuralAdaptiveBias
    print("\n2. Testing NaiveNeuralAdaptiveBias:")
    naive_nab = NaiveNeuralAdaptiveBias(embed_dim=embed_dim, use_duration_matrix=True)
    result_naive = naive_nab(coords, cost_mat, duration_mat)
    print(f"   Output shape: {result_naive.shape}")
    print(f"   Output range: [{result_naive.min():.3f}, {result_naive.max():.3f}]")
    
    # Test AttnFreeNet with different NAB types
    print("\n3. Testing AttnFreeNet with different NAB types:")
    
    # Test with GatingNeuralAdaptiveBias
    print("   3.1. AttnFreeNet with gating NAB:")
    attn_net_gating = AttnFreeNet(
        embed_dim=embed_dim,
        num_layers=2,
        nab_type="gating",
        use_duration_matrix=True
    )
    
    row_emb = torch.randn(batch_size, num_nodes, embed_dim)
    col_emb = torch.randn(batch_size, num_nodes, embed_dim)
    
    row_out_gating, col_out_gating = attn_net_gating(
        row_emb, col_emb, cost_mat, coords, duration_mat
    )
    print(f"       Row output shape: {row_out_gating.shape}")
    print(f"       Col output shape: {col_out_gating.shape}")
    
    # Test with NaiveNeuralAdaptiveBias
    print("   3.2. AttnFreeNet with naive NAB:")
    attn_net_naive = AttnFreeNet(
        embed_dim=embed_dim,
        num_layers=2,
        nab_type="naive",
        use_duration_matrix=True
    )
    
    row_out_naive, col_out_naive = attn_net_naive(
        row_emb, col_emb, cost_mat, coords, duration_mat
    )
    print(f"       Row output shape: {row_out_naive.shape}")
    print(f"       Col output shape: {col_out_naive.shape}")
    
    print("\n4. Test parameter counts:")
    gating_params = sum(p.numel() for p in attn_net_gating.parameters())
    naive_params = sum(p.numel() for p in attn_net_naive.parameters())
    print(f"   Gating NAB parameters: {gating_params:,}")
    print(f"   Naive NAB parameters: {naive_params:,}")
    print(f"   Parameter difference: {gating_params - naive_params:,}")
    
    print("\n✅ All tests passed successfully!")

def test_error_handling():
    """Test error handling for invalid NAB types"""
    print("\n5. Testing error handling:")
    
    try:
        attn_net_invalid = AttnFreeNet(
            embed_dim=128,
            nab_type="invalid_type"
        )
        print("   ❌ Error: Should have raised ValueError for invalid nab_type")
    except ValueError as e:
        print(f"   ✅ Correctly caught error: {e}")

if __name__ == "__main__":
    test_neural_adaptive_bias_types()
    test_error_handling()
    print("\n🎉 All tests completed!") 