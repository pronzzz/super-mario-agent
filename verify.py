"""
Verification script to test the implementation without training.
Tests basic functionality of all components.
"""

import sys
import numpy as np
import torch

print("=" * 60)
print("SUPER MARIO RL AGENT - VERIFICATION SCRIPT")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from src.wrappers import SkipFrame, GrayScaleResize, FrameStack, ActionHistoryWrapper
    from src.model import RainbowDQN, SpatialTransformerNetwork, NoisyLinear
    from src.replay import PrioritizedReplayBuffer, SumTree
    from src.agent import MarioAgent
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Model forward pass
print("\n[2/6] Testing model architecture...")
try:
    device = torch.device('cpu')
    num_actions = 7
    batch_size = 2
    
    model = RainbowDQN(num_actions=num_actions).to(device)
    
    # Create dummy inputs
    frames = torch.randn(batch_size, 4, 84, 84).to(device) / 255.0
    action_history = torch.randint(0, num_actions, (batch_size, 8)).to(device)
    
    # Forward pass
    q_values, q_dist = model(frames, action_history)
    
    assert q_values.shape == (batch_size, num_actions), f"Q-values shape mismatch: {q_values.shape}"
    assert q_dist.shape == (batch_size, num_actions, 51), f"Q-dist shape mismatch: {q_dist.shape}"
    
    print(f"✓ Model forward pass successful")
    print(f"  - Q-values shape: {q_values.shape}")
    print(f"  - Q-distribution shape: {q_dist.shape}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    raise

# Test 3: STN module
print("\n[3/6] Testing Spatial Transformer Network...")
try:
    stn = SpatialTransformerNetwork(in_channels=4).to(device)
    frames = torch.randn(batch_size, 4, 84, 84).to(device)
    transformed = stn(frames)
    
    assert transformed.shape == frames.shape, f"STN shape mismatch: {transformed.shape}"
    print(f"✓ STN test successful")
    print(f"  - Input shape: {frames.shape}")
    print(f"  - Output shape: {transformed.shape}")
except Exception as e:
    print(f"✗ STN test failed: {e}")
    raise

# Test 4: Noisy layers
print("\n[4/6] Testing Noisy Linear layers...")
try:
    noisy_layer = NoisyLinear(128, 64).to(device)
    x = torch.randn(batch_size, 128).to(device)
    
    # Training mode (with noise)
    noisy_layer.train()
    noisy_layer.reset_noise()
    out1 = noisy_layer(x)
    
    # Eval mode (deterministic)
    noisy_layer.eval()
    out2 = noisy_layer(x)
    
    print(f"✓ Noisy layer test successful")
    print(f"  - Output shape: {out1.shape}")
except Exception as e:
    print(f"✗ Noisy layer test failed: {e}")
    raise

# Test 5: Replay buffer
print("\n[5/6] Testing Prioritized Replay Buffer...")
try:
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
    
    # Add some transitions
    for i in range(100):
        frames = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        action_history = np.random.randint(0, 7, 8, dtype=np.int32)
        action = np.random.randint(0, 7)
        reward = np.random.randn()
        next_frames = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        next_action_history = np.random.randint(0, 7, 8, dtype=np.int32)
        done = np.random.rand() > 0.9
        
        buffer.push(frames, action_history, action, reward, 
                   next_frames, next_action_history, done)
    
    # Sample batch
    batch, indices, weights = buffer.sample(32)
    
    assert len(batch) == 32, f"Batch size mismatch: {len(batch)}"
    assert len(indices) == 32, f"Indices size mismatch: {len(indices)}"
    assert len(weights) == 32, f"Weights size mismatch: {len(weights)}"
    
    # Update priorities
    priorities = np.random.rand(32)
    buffer.update_priorities(indices, priorities)
    
    print(f"✓ Replay buffer test successful")
    print(f"  - Buffer size: {len(buffer)}")
    print(f"  - Batch size: {len(batch)}")
    print(f"  - Sample weights: min={weights.min():.3f}, max={weights.max():.3f}")
except Exception as e:
    print(f"✗ Replay buffer test failed: {e}")
    raise

# Test 6: Agent initialization
print("\n[6/6] Testing Agent initialization...")
try:
    agent = MarioAgent(num_actions=7, device='cpu')
    
    # Test action selection
    frames = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action_history = np.random.randint(0, 7, 8, dtype=np.int32)
    
    action = agent.act(frames, action_history)
    
    assert isinstance(action, int), f"Action should be int, got {type(action)}"
    assert 0 <= action < 7, f"Action out of range: {action}"
    
    print(f"✓ Agent test successful")
    print(f"  - Selected action: {action}")
    print(f"  - Device: {agent.device}")
    print(f"  - Replay buffer capacity: {agent.memory.capacity}")
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    raise

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe implementation is ready to use.")
print("\nTo start training:")
print("  1. Install dependencies: pip install -r requirements.txt")
print("  2. Run training: python -m src.train")
print("  3. Monitor progress: tensorboard --logdir=mario_runs")
print("=" * 60)
