#!/usr/bin/env python3
"""PPO training for digital goldfish. Usage: python -m training.train"""

import argparse
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
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path

    with open(path) as f:
        return yaml.safe_load(f)


def make_env(config: dict, num_envs: int = None) -> VecFishEnv:
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
    """Compute Generalized Advantage Estimation."""
    num_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(next_value)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        if dones.ndim == 2 and rewards.ndim == 3:
            next_non_terminal = 1.0 - dones[t][:, np.newaxis]
        else:
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def train(config: dict, args: argparse.Namespace):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    num_envs = args.num_envs or config.get("num_envs", 64)
    total_timesteps = args.total_timesteps or config["training"]["total_timesteps"]

    env = make_env(config, num_envs)
    num_fish = env.num_fish
    single_obs_dim = env.single_obs_dim
    single_action_dim = env.single_action_dim
    print(f"Created {num_envs} parallel environments with {num_fish} fish each")
    print(f"Per-fish obs dim: {single_obs_dim}, action dim: {single_action_dim}")

    policy_config = config.get("policy", {})
    policy = FishPolicy(**policy_config).to(device)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    ppo_config = config.get("ppo", {})
    lr = ppo_config.get("learning_rate", 3e-4)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

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

    samples_per_update = num_envs * num_fish * num_steps
    minibatch_size = samples_per_update // num_minibatches

    env_steps_per_update = num_envs * num_steps
    num_updates = total_timesteps // env_steps_per_update
    print(f"Total updates: {num_updates}")
    print(f"Samples per update: {samples_per_update} ({num_envs} envs x {num_fish} fish x {num_steps} steps)")
    print(f"Minibatch size: {minibatch_size}")

    obs_buffer = np.zeros((num_steps, num_envs, num_fish, single_obs_dim), dtype=np.float32)
    actions_buffer = np.zeros((num_steps, num_envs, num_fish, single_action_dim), dtype=np.float32)
    log_probs_buffer = np.zeros((num_steps, num_envs, num_fish), dtype=np.float32)
    values_buffer = np.zeros((num_steps, num_envs, num_fish), dtype=np.float32)
    rewards_buffer = np.zeros((num_steps, num_envs, num_fish), dtype=np.float32)
    dones_buffer = np.zeros((num_steps, num_envs), dtype=np.float32)

    obs_flat, _ = env.reset()
    global_step = 0
    start_time = time.time()

    log_interval = config["logging"].get("log_interval", 10)
    save_interval = config["logging"].get("save_interval", 100)
    checkpoint_dir = Path(config["paths"].get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(exist_ok=True)

    episode_rewards = []

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = lr * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        for step in range(num_steps):
            global_step += num_envs

            obs = obs_flat.reshape(num_envs, num_fish, single_obs_dim)
            obs_buffer[step] = obs

            obs_for_policy = obs.reshape(num_envs * num_fish, single_obs_dim)
            obs_tensor = torch.from_numpy(obs_for_policy).to(device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action(obs_tensor)

            action_np = action.cpu().numpy().reshape(num_envs, num_fish, single_action_dim)
            log_prob_np = log_prob.cpu().numpy().reshape(num_envs, num_fish)
            value_np = value.cpu().numpy().squeeze(-1).reshape(num_envs, num_fish)

            actions_buffer[step] = action_np
            log_probs_buffer[step] = log_prob_np
            values_buffer[step] = value_np

            action_for_env = action_np.reshape(num_envs, num_fish * single_action_dim)

            obs_flat, reward, terminated, truncated, info = env.step(action_for_env)
            done = terminated | truncated

            rewards_buffer[step] = reward[:, np.newaxis] / num_fish
            dones_buffer[step] = done

            if reward.sum() > 0:
                episode_rewards.extend([r for r in reward if r > 0])

        with torch.no_grad():
            obs = obs_flat.reshape(num_envs, num_fish, single_obs_dim)
            obs_for_policy = obs.reshape(num_envs * num_fish, single_obs_dim)
            next_value = policy.get_value(torch.from_numpy(obs_for_policy).to(device))
            next_value = next_value.cpu().numpy().squeeze(-1).reshape(num_envs, num_fish)

        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer, next_value, gamma, gae_lambda
        )

        b_obs = obs_buffer.reshape(-1, single_obs_dim)
        b_actions = actions_buffer.reshape(-1, single_action_dim)
        b_log_probs = log_probs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(samples_per_update)
        clip_fracs = []
        pg_losses = []
        vf_losses = []
        ent_losses = []

        for _ in range(update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, samples_per_update, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = torch.from_numpy(b_obs[mb_inds]).to(device)
                mb_actions = torch.from_numpy(b_actions[mb_inds]).to(device)
                mb_log_probs = torch.from_numpy(b_log_probs[mb_inds]).to(device)
                mb_advantages = torch.from_numpy(b_advantages[mb_inds]).to(device)
                mb_returns = torch.from_numpy(b_returns[mb_inds]).to(device)

                new_log_prob, entropy, new_value = policy.evaluate_actions(mb_obs, mb_actions)
                new_value = new_value.squeeze(-1)

                log_ratio = new_log_prob - mb_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(pg_loss.item())

                vf_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                vf_losses.append(vf_loss.item())

                ent_loss = entropy.mean()
                ent_losses.append(ent_loss.item())

                loss = pg_loss - ent_coef * ent_loss + vf_coef * vf_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
