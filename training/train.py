#!/usr/bin/env python3
"""
Training script for digital goldfish using PPO.

Usage:
    python -m training.train
    python -m training.train --config training/configs/default.yaml
    python -m training.train --num-envs 128 --total-timesteps 2000000
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from training.envs import VecFishEnv
from training.models import FishPolicy


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path

    with open(path) as f:
        return yaml.safe_load(f)


def make_env(config: dict, num_envs: int = None) -> VecFishEnv:
    """Create vectorized environment."""
    env_config = config.get("env", {})
    n = num_envs or config.get("num_envs", 64)
    return VecFishEnv(num_envs=n, config=env_config)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def train(config: dict, args: argparse.Namespace):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config with command line args
    num_envs = args.num_envs or config.get("num_envs", 64)
    total_timesteps = args.total_timesteps or config["training"]["total_timesteps"]

    # Create environment
    env = make_env(config, num_envs)
    print(f"Created {num_envs} parallel environments")

    # Create policy
    policy_config = config.get("policy", {})
    policy = FishPolicy(**policy_config).to(device)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Optimizer
    ppo_config = config.get("ppo", {})
    lr = ppo_config.get("learning_rate", 3e-4)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Training hyperparameters
    num_steps = config["training"]["num_steps"]
    num_minibatches = config["training"]["num_minibatches"]
    update_epochs = config["training"]["update_epochs"]
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_coef = ppo_config.get("clip_coef", 0.2)
    ent_coef = ppo_config.get("ent_coef", 0.01)
    vf_coef = ppo_config.get("vf_coef", 0.5)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    anneal_lr = config["training"].get("anneal_lr", True)

    # Batch sizes
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    # Calculate number of updates
    num_updates = total_timesteps // batch_size
    print(f"Total updates: {num_updates}")
    print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")

    # Storage for rollout
    obs_buffer = np.zeros((num_steps, num_envs, env.OBS_DIM), dtype=np.float32)
    actions_buffer = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
    rewards_buffer = np.zeros((num_steps, num_envs), dtype=np.float32)
    dones_buffer = np.zeros((num_steps, num_envs), dtype=np.float32)
    values_buffer = np.zeros((num_steps, num_envs), dtype=np.float32)
    log_probs_buffer = np.zeros((num_steps, num_envs), dtype=np.float32)

    # Initialize
    obs, _ = env.reset()
    global_step = 0
    start_time = time.time()

    # Logging
    log_interval = config["logging"].get("log_interval", 10)
    save_interval = config["logging"].get("save_interval", 100)
    checkpoint_dir = Path(config["paths"].get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(exist_ok=True)

    episode_rewards = []
    episode_lengths = []

    for update in range(1, num_updates + 1):
        # Anneal learning rate
        if anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = lr * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        # Collect rollout
        for step in range(num_steps):
            global_step += num_envs

            obs_buffer[step] = obs
            obs_tensor = torch.from_numpy(obs).to(device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action(obs_tensor)
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy().squeeze(-1)

            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            rewards_buffer[step] = reward
            dones_buffer[step] = done

            # Track episode stats
            if reward.sum() > 0:
                episode_rewards.extend([r for r in reward if r > 0])

        # Compute GAE
        with torch.no_grad():
            next_value = (
                policy.get_value(torch.from_numpy(obs).to(device)).cpu().numpy().squeeze(-1)
            )

        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer, next_value, gamma, gae_lambda
        )

        # Flatten batches
        b_obs = obs_buffer.reshape(-1, env.OBS_DIM)
        b_actions = actions_buffer.reshape(-1, 2)
        b_log_probs = log_probs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        b_inds = np.arange(batch_size)
        clip_fracs = []
        pg_losses = []
        vf_losses = []
        ent_losses = []

        for _ in range(update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = torch.from_numpy(b_obs[mb_inds]).to(device)
                mb_actions = torch.from_numpy(b_actions[mb_inds]).to(device)
                mb_log_probs = torch.from_numpy(b_log_probs[mb_inds]).to(device)
                mb_advantages = torch.from_numpy(b_advantages[mb_inds]).to(device)
                mb_returns = torch.from_numpy(b_returns[mb_inds]).to(device)

                # Forward pass
                new_log_prob, entropy, new_value = policy.evaluate_actions(mb_obs, mb_actions)
                new_value = new_value.squeeze(-1)

                # Policy loss
                log_ratio = new_log_prob - mb_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(pg_loss.item())

                # Value loss
                vf_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                vf_losses.append(vf_loss.item())

                # Entropy loss
                ent_loss = entropy.mean()
                ent_losses.append(ent_loss.item())

                # Total loss
                loss = pg_loss - ent_coef * ent_loss + vf_coef * vf_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        if update % log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed

            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            print(
                f"Update {update:5d} | "
                f"Steps {global_step:8d} | "
                f"FPS {fps:6.0f} | "
                f"Reward {avg_reward:.3f} | "
                f"PG {np.mean(pg_losses):.4f} | "
                f"VF {np.mean(vf_losses):.4f} | "
                f"Ent {np.mean(ent_losses):.4f} | "
                f"Clip {np.mean(clip_fracs):.3f}"
            )

        # Save checkpoint
        if update % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"policy_{update}.pt"
            torch.save(
                {
                    "update": update,
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = checkpoint_dir / "policy_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete! Final model saved to: {final_path}")

    return policy


def main():
    parser = argparse.ArgumentParser(description="Train digital goldfish with PPO")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Train
    train(config, args)


if __name__ == "__main__":
    main()
