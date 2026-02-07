#!/usr/bin/env python3
"""
Live visualization of trained fish policy using pygame.
"""

import torch
import numpy as np
import pygame
import sys
import math
from pathlib import Path

from training.envs import FishEnv
from training.models import FishPolicy


def run_visualization(checkpoint_path: str = "checkpoints/policy_final.pt"):
    pygame.init()

    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Digital Goldfish - Live View")
    clock = pygame.time.Clock()

    BG_COLOR = (20, 40, 60)
    FISH_COLOR = (255, 180, 50)
    FISH_OUTLINE = (200, 120, 30)
    FOOD_COLOR = (100, 255, 100)
    TEXT_COLOR = (255, 255, 255)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    policy = FishPolicy()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy = policy.to(device)
    policy.eval()

    env = FishEnv(config={
        'width': WIDTH,
        'height': HEIGHT,
        'max_steps': 10000,
        'initial_food': 10,
        'food_spawn_rate': 0.02,
    })

    obs, _ = env.reset()
    food_eaten = 0
    episode = 1
    step = 0

    # Fin animation state (auto-computed from actions)
    tail_phase = 0.0
    body_curve = 0.0
    dt = 1.0 / 60.0  # 60 FPS

    font = pygame.font.Font(None, 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    food_eaten = 0
                    episode += 1
                    step = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if env.food_count < env.max_food:
                    env.food[env.food_count] = [mx, my, 0.0]
                    env.food_count += 1

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            action, _, _, _ = policy.get_action(obs_tensor, deterministic=False)
            action = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        if reward > 0.5:
            food_eaten += int(reward)

        if terminated or truncated:
            obs, _ = env.reset()
            episode += 1
            food_eaten = 0
            step = 0
            tail_phase = 0.0
            body_curve = 0.0

        # Compute fin animation from 3-action API (speed, direction, urgency)
        # Matches fish.c logic
        speed = max(0.0, min(1.0, action[0]))
        direction = max(-1.0, min(1.0, action[1]))
        urgency = max(0.0, min(1.0, action[2])) if len(action) > 2 else 0.5

        tail_freq = 0.3 + urgency * 0.2  # 0.3-0.5 Hz
        tail_amplitude = speed * 0.8
        tail_phase += tail_freq * 2.0 * math.pi * dt
        if tail_phase > 2.0 * math.pi:
            tail_phase -= 2.0 * math.pi

        target_curve = direction * 0.4
        body_curve += (target_curve - body_curve) * 5.0 * dt

        base_pec = 0.3 - speed * 0.2
        left_pec = base_pec + direction * 0.3
        right_pec = base_pec - direction * 0.3

        screen.fill(BG_COLOR)

        for i in range(env.food_count):
            fx, fy = int(env.food[i, 0]), int(env.food[i, 1])
            age = env.food[i, 2]
            # Food glows brighter with age (easier to detect)
            glow = min(1.0, 0.5 + age * 0.1)
            color = (int(100 * glow), int(255 * glow), int(100 * glow))
            pygame.draw.circle(screen, color, (fx, fy), 6)
            pygame.draw.circle(screen, (50, 150, 50), (fx, fy), 6, 1)

        fx, fy = int(env.pos[0]), int(env.pos[1])
        angle = env.angle

        fish_length = 80
        fish_width = 32

        cos_a, sin_a = math.cos(angle), math.sin(angle)

        nose = (fx + cos_a * fish_length, fy + sin_a * fish_length)
        tail = (fx - cos_a * fish_length * 0.7, fy - sin_a * fish_length * 0.7)
        left = (fx - sin_a * fish_width, fy + cos_a * fish_width)
        right = (fx + sin_a * fish_width, fy - cos_a * fish_width)

        body_points = [nose, right, tail, left]
        pygame.draw.polygon(screen, FISH_COLOR, body_points)
        pygame.draw.polygon(screen, FISH_OUTLINE, body_points, 2)

        tail_swing = math.sin(tail_phase) * 0.5 * tail_amplitude
        tail_angle = angle + math.pi + tail_swing
        tail_cos, tail_sin = math.cos(tail_angle), math.sin(tail_angle)
        tail_tip = (tail[0] + tail_cos * 40, tail[1] + tail_sin * 40)
        tail_left = (tail[0] - tail_sin * 21, tail[1] + tail_cos * 21)
        tail_right = (tail[0] + tail_sin * 21, tail[1] - tail_cos * 21)
        pygame.draw.polygon(screen, FISH_COLOR, [tail, tail_left, tail_tip, tail_right])

        eye_offset = 21
        eye_x = fx + cos_a * eye_offset - sin_a * 11
        eye_y = fy + sin_a * eye_offset + cos_a * 11
        pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 8)
        pygame.draw.circle(screen, (0, 0, 0), (int(eye_x), int(eye_y)), 3)

        pec_offset = 13
        for side, pec_val in [(-1, left_pec), (1, right_pec)]:
            pec_x = fx - cos_a * pec_offset + sin_a * fish_width * 0.6 * side
            pec_y = fy - sin_a * pec_offset - cos_a * fish_width * 0.6 * side
            fin_angle = angle + side * (1.5 - pec_val * 0.5)
            fin_cos, fin_sin = math.cos(fin_angle), math.sin(fin_angle)
            fin_tip = (pec_x + fin_cos * 27, pec_y + fin_sin * 27)
            pygame.draw.line(screen, FISH_OUTLINE, (int(pec_x), int(pec_y)),
                           (int(fin_tip[0]), int(fin_tip[1])), 3)

        hud_texts = [
            f"Episode: {episode}  Step: {step}  Food: {food_eaten}",
            f"Speed: {speed:.2f}  Dir: {direction:+.2f}  Urgency: {urgency:.2f}",
            "Click to add food | R to reset | ESC to quit"
        ]
        for i, text in enumerate(hud_texts):
            surf = font.render(text, True, TEXT_COLOR)
            screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/policy_final.pt')
    args = parser.parse_args()
    run_visualization(args.checkpoint)
