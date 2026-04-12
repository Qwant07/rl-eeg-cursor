"""Unified PPO training script for BCI cursor control.

Trains Naïve PPO or Constrained PPO on the cursor environment with
configurable noise model, curriculum, and latency.

Usage:
    # Naïve PPO on S05
    python -m src.agents.train_ppo --subject S05 --mode naive --total_steps 500000

    # Constrained PPO on S05
    python -m src.agents.train_ppo --subject S05 --mode constrained --total_steps 500000

    # With curriculum (recommended for S01)
    python -m src.agents.train_ppo --subject S01 --mode constrained --curriculum --total_steps 500000
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.cursor_env import CursorEnv
from src.envs.noise_model import DecoderNoiseModel
from src.envs.curriculum import CurriculumWrapper
from src.agents.constrained_wrapper import ConstrainedRewardWrapper


class MetricsCallback(BaseCallback):
    """Logs custom metrics: success rate, avg steps, penalties."""

    def __init__(self, eval_env, eval_freq=10000, n_eval=50, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            results = self._evaluate()
            self.eval_results.append({
                "timestep": self.num_timesteps,
                **results,
            })
            if self.verbose:
                print(
                    f"  [eval@{self.num_timesteps}] "
                    f"success={results['success_rate']:.0%} "
                    f"avg_steps={results['avg_steps']:.0f}"
                )
            self.logger.record("eval/success_rate", results["success_rate"])
            self.logger.record("eval/avg_steps", results["avg_steps"])
        return True

    def _evaluate(self) -> dict:
        successes, steps_list = 0, []
        for ep in range(self.n_eval):
            obs, _ = self.eval_env.reset(seed=ep + 10000)
            for step in range(self.eval_env.unwrapped.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                if terminated:
                    successes += 1
                    steps_list.append(step + 1)
                    break
                if truncated:
                    steps_list.append(step + 1)
                    break
        return {
            "success_rate": successes / self.n_eval,
            "successes": successes,
            "avg_steps": float(np.mean(steps_list)) if steps_list else 0,
        }


def make_env(subject, latency_steps=0, mode="naive",
             lambda_smooth=0.1, lambda_zero=0.05,
             curriculum=False, total_steps=500000):
    """Build the training environment."""
    noise = DecoderNoiseModel.from_subject(subject, rng=np.random.default_rng())
    env = CursorEnv(
        noise_model=noise,
        dt=0.1,
        vel_scale=0.5,
        max_steps=200,
        latency_steps=latency_steps,
        target_radius=0.05,
        dwell_steps=4,
        success_bonus=10.0,
    )

    if curriculum:
        env = CurriculumWrapper(env, total_steps=total_steps)

    if mode == "constrained":
        env = ConstrainedRewardWrapper(
            env,
            lambda_smooth=lambda_smooth,
            lambda_zero=lambda_zero,
        )

    return env


def make_eval_env(subject, latency_steps=0):
    """Build evaluation env (no curriculum, no constraints — raw performance)."""
    noise = DecoderNoiseModel.from_subject(subject, rng=np.random.default_rng(99))
    return CursorEnv(
        noise_model=noise,
        dt=0.1,
        vel_scale=0.5,
        max_steps=200,
        latency_steps=latency_steps,
        target_radius=0.05,
        dwell_steps=4,
        success_bonus=10.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO for BCI cursor control")
    parser.add_argument("--subject", type=str, required=True, choices=["S01", "S05"])
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "constrained"])
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency_steps", type=int, default=0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_zero", type=float, default=0.05)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    # PPO hyperparams
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    args = parser.parse_args()

    # Build envs
    train_env = make_env(
        args.subject, args.latency_steps, args.mode,
        args.lambda_smooth, args.lambda_zero,
        args.curriculum, args.total_steps,
    )
    train_env = Monitor(train_env)

    eval_env = make_eval_env(args.subject, args.latency_steps)

    # Build PPO
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        device=args.device,
        tensorboard_log=os.path.join(args.out_dir, args.subject, "tb_logs"),
    )

    # Callbacks
    metrics_cb = MetricsCallback(
        eval_env, eval_freq=args.eval_freq, n_eval=50, verbose=1,
    )

    # Train
    run_name = f"{args.mode}_seed{args.seed}"
    if args.curriculum:
        run_name += "_curriculum"
    print(f"\n{'='*60}")
    print(f"Training: {args.subject} / {run_name}")
    print(f"Steps: {args.total_steps}, Device: {args.device}")
    if args.mode == "constrained":
        print(f"λ_smooth={args.lambda_smooth}, λ_zero={args.lambda_zero}")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=metrics_cb,
        tb_log_name=run_name,
    )
    elapsed = time.time() - t0

    # Final evaluation
    final = metrics_cb._evaluate()
    print(f"\nTraining finished in {elapsed:.0f}s")
    print(f"Final: success={final['success_rate']:.0%}, avg_steps={final['avg_steps']:.0f}")

    # Save
    out_dir = Path(args.out_dir) / args.subject / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(out_dir / "ppo_model"))
    print(f"Saved model → {out_dir / 'ppo_model.zip'}")

    results = {
        "subject": args.subject,
        "mode": args.mode,
        "seed": args.seed,
        "curriculum": args.curriculum,
        "total_steps": args.total_steps,
        "latency_steps": args.latency_steps,
        "lambda_smooth": args.lambda_smooth if args.mode == "constrained" else None,
        "lambda_zero": args.lambda_zero if args.mode == "constrained" else None,
        "training_time_s": elapsed,
        "final_eval": final,
        "eval_history": metrics_cb.eval_results,
        "hyperparams": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
