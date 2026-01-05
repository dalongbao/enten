#!/usr/bin/env python3
"""
Test a trained policy with simple ASCII visualization.
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from training.envs import FishEnv
from training.models import FishPolicy


def render_ascii(env, width=80, height=24):
    """Simple ASCII render of the environment."""
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Scale positions
    scale_x = width / env.width
    scale_y = height / env.height

    # Draw food as dots
    for i in range(env.food_count):
        fx = int(env.food[i, 0] * scale_x)
        fy = int(env.food[i, 1] * scale_y)
        if 0 <= fx < width and 0 <= fy < height:
            grid[fy][fx] = '·'

    # Draw fish as arrow based on angle
    px = int(env.pos[0] * scale_x)
    py = int(env.pos[1] * scale_y)

    # Arrow characters based on angle
    angle = env.angle % (2 * np.pi)
    if angle < np.pi / 8 or angle >= 15 * np.pi / 8:
        fish_char = '→'
    elif angle < 3 * np.pi / 8:
        fish_char = '↘'
    elif angle < 5 * np.pi / 8:
        fish_char = '↓'
    elif angle < 7 * np.pi / 8:
        fish_char = '↙'
    elif angle < 9 * np.pi / 8:
        fish_char = '←'
    elif angle < 11 * np.pi / 8:
        fish_char = '↖'
    elif angle < 13 * np.pi / 8:
        fish_char = '↑'
    else:
        fish_char = '↗'

    if 0 <= px < width and 0 <= py < height:
        grid[py][px] = fish_char

    # Draw border and convert to string
    border = '─' * width
    output = ['┌' + border + '┐']
    for row in grid:
        output.append('│' + ''.join(row) + '│')
    output.append('└' + border + '┘')

    return '\n'.join(output)


def test_policy(checkpoint_path: str, num_episodes: int = 3, render: bool = True):
    """Test a trained policy."""
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load policy
    policy = FishPolicy()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded checkpoint from update {checkpoint.get('update', '?')}")
    else:
        policy.load_state_dict(checkpoint)
        print("Loaded final policy weights")

    policy = policy.to(device)
    policy.eval()

    # Create environment
    env = FishEnv(config={
        'width': 800,
        'height': 600,
        'max_steps': 500,
        'initial_food': 10,
        'food_spawn_rate': 0.03,
    })

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        food_eaten = 0

        print(f"\n=== Episode {ep + 1} ===")

        for step in range(500):
            # Get action
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, _, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action = action.cpu().numpy()[0]

            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if reward > 0.5:  # Food eaten
                food_eaten += int(reward)

            # Render
            if render and step % 10 == 0:
                print('\033[2J\033[H')  # Clear screen
                print(f"Episode {ep + 1} | Step {step} | Food eaten: {food_eaten} | Reward: {total_reward:.2f}")
                print(f"Action: tail={action[0]:.2f} curve={action[1]:.2f} L={action[2]:.2f} R={action[3]:.2f}")
                print(render_ascii(env))
                time.sleep(0.05)

            if terminated or truncated:
                break

        print(f"\nEpisode {ep + 1} finished: {food_eaten} food eaten, total reward: {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/policy_100.pt')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()

    test_policy(args.checkpoint, args.episodes, not args.no_render)


if __name__ == '__main__':
    main()
