"""
Test script for AMP components.

This script verifies that all AMP components can be initialized and
work together correctly.
"""

import torch
import numpy as np
from discriminator import Discriminator
from replay_buffer import ReplayBuffer
from normalizer import Normalizer
from demo_loader import DemoLoader


def test_discriminator():
    """Test discriminator initialization and forward pass."""
    print("Testing Discriminator...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc = Discriminator(obs_dim=8, hidden_dims=[256, 256]).to(device)
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, 8, device=device)
    logits = disc(obs)
    
    assert logits.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {logits.shape}"
    
    # Test methods
    logit_weights = disc.get_logit_weights()
    all_weights = disc.get_all_weights()
    
    assert logit_weights.dim() == 1, "Logit weights should be 1D"
    assert all_weights.dim() == 1, "All weights should be 1D"
    
    print("✓ Discriminator tests passed")


def test_normalizer():
    """Test normalizer recording and normalization."""
    print("\nTesting Normalizer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm = Normalizer(shape=(8,), clip=10.0, device=device)
    
    # Record some data
    data1 = torch.randn(100, 8, device=device)
    data2 = torch.randn(100, 8, device=device)
    
    norm.record(data1)
    norm.record(data2)
    norm.update()
    
    # Test normalization
    test_data = torch.randn(50, 8, device=device)
    normalized = norm.normalize(test_data)
    
    assert normalized.shape == test_data.shape, "Shape mismatch after normalization"
    assert torch.all(normalized >= -10) and torch.all(normalized <= 10), "Normalization clipping failed"
    
    # Test denormalization
    denormalized = norm.denormalize(normalized)
    assert denormalized.shape == test_data.shape, "Shape mismatch after denormalization"
    
    print("✓ Normalizer tests passed")


def test_replay_buffer():
    """Test replay buffer push and sample."""
    print("\nTesting ReplayBuffer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer = ReplayBuffer(capacity=1000, obs_shape=(8,), device=device)
    
    # Test push
    data = torch.randn(100, 8, device=device)
    buffer.push(data)
    
    assert len(buffer) == 100, f"Expected buffer size 100, got {len(buffer)}"
    
    # Test sample
    samples = buffer.sample(50)
    assert samples.shape == (50, 8), f"Expected shape (50, 8), got {samples.shape}"
    
    # Test is_full
    assert not buffer.is_full(), "Buffer should not be full"
    
    # Fill buffer
    buffer.push(torch.randn(1000, 8, device=device))
    assert buffer.is_full(), "Buffer should be full"
    
    print("✓ ReplayBuffer tests passed")


def test_demo_loader():
    """Test demo loader (requires demo data file)."""
    print("\nTesting DemoLoader...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo_path = "scripts/smooth_policy/amp_data/amp_dataset.pt"
    
    try:
        loader = DemoLoader(demo_path, device=device)
        
        # Test sample
        samples = loader.sample(64)
        assert samples.shape[0] == 64, f"Expected 64 samples, got {samples.shape[0]}"
        assert samples.shape[1] == 8, f"Expected 8D observations, got {samples.shape[1]}D"
        
        # Test get_all
        all_demos = loader.get_all()
        assert len(all_demos) == len(loader), "Mismatch in data length"
        
        print("✓ DemoLoader tests passed")
        
    except FileNotFoundError as e:
        print(f"⚠ DemoLoader test skipped: {e}")
        print("  Run prepare_amp_dataset.py first to create demo data")


def test_integration():
    """Test all components working together."""
    print("\nTesting Integration...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize components
    disc = Discriminator(obs_dim=8).to(device)
    norm = Normalizer(shape=(8,), clip=10.0, device=device)
    buffer = ReplayBuffer(capacity=1000, obs_shape=(8,), device=device)
    
    # Simulate training loop
    batch_size = 128
    
    # Generate fake agent data
    agent_data = torch.randn(batch_size, 8, device=device)
    
    # Normalize
    norm.record(agent_data)
    norm.update()
    normalized_agent = norm.normalize(agent_data)
    
    # Forward through discriminator
    logits = disc(normalized_agent)
    assert logits.shape == (batch_size, 1), "Discriminator output shape mismatch"
    
    # Store in replay buffer
    buffer.push(agent_data)
    assert len(buffer) == batch_size, "Buffer size mismatch"
    
    # Sample from buffer
    samples = buffer.sample(64)
    normalized_samples = norm.normalize(samples)
    logits_samples = disc(normalized_samples)
    assert logits_samples.shape == (64, 1), "Sampled discriminator output shape mismatch"
    
    print("✓ Integration tests passed")


if __name__ == "__main__":
    print("="*80)
    print("Running AMP Component Tests")
    print("="*80)
    
    test_discriminator()
    test_normalizer()
    test_replay_buffer()
    test_demo_loader()
    test_integration()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
