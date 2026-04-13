"""Evaluate BCI cursor control policies with formal metrics.

Runs policies (PPO, BC, proportional) across many episodes, computes
Fitts' throughput, path efficiency, smoothness, and Shannon ITR.
Supports multi-seed aggregation with statistical tests.

Usage:
    # Evaluate a saved PPO model
    python -m src.evaluation.run_eval --policy ppo --model_path results/S05/naive_seed0/ppo_model.zip --subject S05

    # Evaluate proportional controller baseline
    python -m src.evaluation.run_eval --policy proportional --subject S05

    # Compare two conditions with paired t-test
    python -m src.evaluation.run_eval --compare results/S05/naive_eval.json results/S05/constrained_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from src.envs.cursor_env import CursorEnv
from src.envs.noise_model import DecoderNoiseModel
from src.evaluation.metrics import (
    compute_episode_metrics,
    aggregate_metrics,
    shannon_itr,
    EpisodeMetrics,
)


def evaluate_policy(
    policy_fn,
    env: CursorEnv,
    n_episodes: int = 100,
    seed_offset: int = 0,
) -> list[EpisodeMetrics]:
    """Run a policy for n_episodes and collect per-episode metrics.

    Args:
        policy_fn: callable(obs) -> action array (2,).
        env: CursorEnv instance (unwrapped, no wrappers).
        n_episodes: number of evaluation episodes.
        seed_offset: base seed for reproducibility.

    Returns:
        List of EpisodeMetrics, one per episode.
    """
    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        start_pos = info["cursor_pos"].copy()
        target_pos = info["target_pos"].copy()

        positions = [start_pos.copy()]
        actions_list = []
        success = False

        for step in range(env.max_steps):
            action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)

            positions.append(info["cursor_pos"].copy())
            actions_list.append(action.copy())

            if terminated:
                success = True
                break
            if truncated:
                break

        cursor_positions = np.array(positions)
        actions_arr = np.array(actions_list) if actions_list else np.zeros((0, 2))

        metrics = compute_episode_metrics(
            cursor_positions=cursor_positions,
            actions=actions_arr,
            start_pos=start_pos,
            target_pos=target_pos,
            target_radius=env.target_radius,
            dt=env.dt,
            success=success,
        )
        episodes.append(metrics)

    return episodes


def proportional_policy(obs: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Proportional controller: full-speed unit vector toward target.

    Matches the controller used during training/baseline evaluation.
    Observations are normalized (cursor_pos / workspace), but the
    direction is scale-invariant so we just use normalized coords.
    """
    cursor = obs[:2]
    target = obs[2:4]
    diff = target - cursor
    dist = np.linalg.norm(diff)
    if dist < 0.01:
        return np.zeros(2, dtype=np.float32)
    return (diff / dist).astype(np.float32)


def make_eval_env(subject: str, latency_steps: int = 0, seed: int = 99) -> CursorEnv:
    """Build evaluation environment."""
    noise = DecoderNoiseModel.from_subject(subject, rng=np.random.default_rng(seed))
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


def compare_conditions(
    metrics_a: list[EpisodeMetrics],
    metrics_b: list[EpisodeMetrics],
    metric_name: str = "action_smoothness",
) -> dict:
    """Compare two conditions with Wilcoxon signed-rank test.

    Uses paired test when episode counts match (same seeds),
    otherwise Mann-Whitney U.

    Args:
        metrics_a: episodes from condition A.
        metrics_b: episodes from condition B.
        metric_name: attribute name on EpisodeMetrics to compare.

    Returns:
        Dict with statistic, p_value, effect_size (Cohen's d).
    """
    vals_a = np.array([getattr(e, metric_name) for e in metrics_a if e.success])
    vals_b = np.array([getattr(e, metric_name) for e in metrics_b if e.success])

    if len(vals_a) < 3 or len(vals_b) < 3:
        return {"error": "Too few successful episodes for statistical test"}

    # Use paired test if same length (same seed episodes)
    if len(vals_a) == len(vals_b):
        stat_result = stats.wilcoxon(vals_a, vals_b)
        test_name = "wilcoxon"
    else:
        stat_result = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        test_name = "mannwhitneyu"

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(vals_a, ddof=1) + np.var(vals_b, ddof=1)) / 2)
    cohens_d = (np.mean(vals_a) - np.mean(vals_b)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": test_name,
        "metric": metric_name,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "cohens_d": float(cohens_d),
        "significant_005": stat_result.pvalue < 0.05,
        "significant_001": stat_result.pvalue < 0.01,
        "n_a": len(vals_a),
        "n_b": len(vals_b),
    }


def multi_seed_aggregate(
    seed_results: list[dict],
) -> dict:
    """Aggregate metrics across multiple seeds.

    Args:
        seed_results: list of aggregate_metrics() dicts, one per seed.

    Returns:
        Dict with cross-seed mean ± std for each metric.
    """
    if not seed_results:
        return {}

    result = {"n_seeds": len(seed_results)}
    keys_to_agg = [k for k in seed_results[0] if k.endswith("_mean") or k == "success_rate"]

    for key in keys_to_agg:
        values = [s[key] for s in seed_results if key in s]
        if values:
            result[f"{key}_across_seeds_mean"] = float(np.mean(values))
            result[f"{key}_across_seeds_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return result


def results_to_json(
    episodes: list[EpisodeMetrics],
    agg: dict,
    config: dict,
) -> dict:
    """Package evaluation results for JSON serialization."""
    return {
        "config": config,
        "aggregate": agg,
        "per_episode": [
            {
                "success": e.success,
                "movement_time": e.movement_time,
                "steps": e.steps,
                "fitts_id": e.fitts_id,
                "fitts_tp": e.fitts_tp,
                "path_efficiency": e.path_efficiency,
                "action_smoothness": e.action_smoothness,
                "mean_action_magnitude": e.mean_action_magnitude,
                "final_distance": e.final_distance,
                "target_distance": e.target_distance,
            }
            for e in episodes
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BCI cursor control policies")
    sub = parser.add_subparsers(dest="command")

    # Evaluate command
    eval_parser = sub.add_parser("eval", help="Evaluate a single policy")
    eval_parser.add_argument("--policy", required=True, choices=["ppo", "proportional"])
    eval_parser.add_argument("--model_path", type=str, help="Path to PPO model .zip")
    eval_parser.add_argument("--subject", required=True, choices=["S01", "S05"])
    eval_parser.add_argument("--n_episodes", type=int, default=100)
    eval_parser.add_argument("--latency_steps", type=int, default=0)
    eval_parser.add_argument("--seed_offset", type=int, default=0)
    eval_parser.add_argument("--out", type=str, help="Output JSON path")

    # Compare command
    cmp_parser = sub.add_parser("compare", help="Compare two evaluation result files")
    cmp_parser.add_argument("file_a", type=str)
    cmp_parser.add_argument("file_b", type=str)
    cmp_parser.add_argument("--metrics", nargs="+",
                            default=["action_smoothness", "movement_time", "path_efficiency", "fitts_tp"])

    args = parser.parse_args()

    if args.command == "eval":
        env = make_eval_env(args.subject, args.latency_steps)

        if args.policy == "proportional":
            policy_fn = proportional_policy
        elif args.policy == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)
            policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]

        print(f"Evaluating {args.policy} on {args.subject} ({args.n_episodes} episodes)...")
        episodes = evaluate_policy(policy_fn, env, args.n_episodes, args.seed_offset)
        agg = aggregate_metrics(episodes)

        # Shannon ITR
        itr = shannon_itr(
            n_targets=8,
            accuracy=agg["success_rate"],
            trial_time=agg.get("movement_time_mean", 0),
        )
        agg["shannon_itr_bpm"] = itr

        # Print summary
        print(f"\n{'='*50}")
        print(f"  {args.policy.upper()} on {args.subject}")
        print(f"{'='*50}")
        print(f"  Success rate:      {agg['success_rate']:.1%}")
        if agg.get("movement_time_mean"):
            print(f"  Movement time:     {agg['movement_time_mean']:.2f} ± {agg['movement_time_ci95']:.2f} s")
            print(f"  Steps:             {agg['steps_mean']:.1f} ± {agg['steps_ci95']:.1f}")
            print(f"  Fitts' throughput: {agg['fitts_tp_mean']:.3f} ± {agg['fitts_tp_ci95']:.3f} bits/s")
            print(f"  Path efficiency:   {agg['path_efficiency_mean']:.3f} ± {agg['path_efficiency_ci95']:.3f}")
            print(f"  Smoothness:        {agg['action_smoothness_mean']:.4f} ± {agg['action_smoothness_ci95']:.4f}")
            print(f"  Shannon ITR:       {itr:.2f} bits/min")
        print()

        config = {
            "policy": args.policy,
            "subject": args.subject,
            "n_episodes": args.n_episodes,
            "latency_steps": args.latency_steps,
            "model_path": args.model_path,
        }
        output = results_to_json(episodes, agg, config)

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Saved → {args.out}")

    elif args.command == "compare":
        with open(args.file_a) as f:
            data_a = json.load(f)
        with open(args.file_b) as f:
            data_b = json.load(f)

        print(f"\nComparing: {args.file_a} vs {args.file_b}")
        print(f"{'='*60}")

        # Reconstruct minimal EpisodeMetrics from JSON for comparison
        for metric_name in args.metrics:
            vals_a = [ep[metric_name] for ep in data_a["per_episode"] if ep["success"]]
            vals_b = [ep[metric_name] for ep in data_b["per_episode"] if ep["success"]]

            if len(vals_a) < 3 or len(vals_b) < 3:
                print(f"  {metric_name}: too few episodes for test")
                continue

            vals_a, vals_b = np.array(vals_a), np.array(vals_b)
            if len(vals_a) == len(vals_b):
                stat_result = stats.wilcoxon(vals_a, vals_b)
                test = "Wilcoxon"
            else:
                stat_result = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                test = "Mann-Whitney"

            sig = "*" if stat_result.pvalue < 0.05 else ""
            sig += "*" if stat_result.pvalue < 0.01 else ""
            sig += "*" if stat_result.pvalue < 0.001 else ""

            print(f"  {metric_name}:")
            print(f"    A: {np.mean(vals_a):.4f} ± {np.std(vals_a, ddof=1):.4f}")
            print(f"    B: {np.mean(vals_b):.4f} ± {np.std(vals_b, ddof=1):.4f}")
            print(f"    {test}: p={stat_result.pvalue:.4f} {sig}")
        print()


if __name__ == "__main__":
    main()
