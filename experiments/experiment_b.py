#!/usr/bin/env python3
"""Experiment B: The Fake Lava Test (Hallucination)

Tests whether the agent can override wrong VLM priors.

Scenario: The room is full of "Red Water" which:
- VLM predicts as dangerous (looks like fire/lava)
- Is actually safe to walk on

Success Metric: Time to "unlearn" the prior
- If it takes 1,000 steps: method fails
- If it takes 10 steps: method works

Conditions:
1. Pure SwiftTD: No VLM (Control)
2. VLM Init + Hallucinating: VLM says red water is dangerous (wrong)
3. Oracle Init: Perfect labels (red water is safe)
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from experiments.runner import ExperimentConfig, ExperimentRunner, run_condition, aggregate_results


def run_experiment_b(
    seeds: list[int] = None,
    n_episodes: int = 100,
    output_dir: str = "data/results/experiment_b",
    verbose: bool = False,
):
    """Run all conditions for Experiment B (Fake Lava)."""
    if seeds is None:
        seeds = list(range(10))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Condition 1: Pure SwiftTD (no VLM)
    print("Running Condition 1: Pure SwiftTD (no VLM)...")
    results['pure_swifttd'] = run_condition(
        name="exp_b_pure_swifttd",
        env_type="fake_lava",
        agent_type="swifttd",
        vlm_type="none",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 2: VLM Init with hallucination (mock)
    print("Running Condition 2: VLM Init (Mock Hallucinating)...")
    results['vlm_hallucinating'] = run_condition(
        name="exp_b_vlm_hallucinating",
        env_type="fake_lava",
        agent_type="swifttd_vlm",
        vlm_type="hallucinating",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 3: Real Moondream VLM (actual hallucination!)
    print("Running Condition 3: Real Moondream VLM...")
    results['moondream_vlm'] = run_condition(
        name="exp_b_moondream",
        env_type="fake_lava",
        agent_type="swifttd_vlm",
        vlm_type="ollama",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 4: Oracle Init (no hallucination)
    print("Running Condition 4: Oracle Init (Perfect)...")
    results['oracle_init'] = run_condition(
        name="exp_b_oracle_init",
        env_type="fake_lava",
        agent_type="swifttd_vlm",
        vlm_type="oracle",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Save aggregated results
    summary = {
        'experiment': 'B',
        'name': 'Fake Lava Test (Hallucination)',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': len(seeds),
            'n_episodes': n_episodes,
        },
        'conditions': {}
    }

    for name, condition_results in results.items():
        summary['conditions'][name] = aggregate_results(condition_results)
        # Save individual run results
        for r in condition_results:
            r.save(str(output_path / f"{r.config.name}.json"))

    # Save summary
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print_summary(summary)

    return summary


def print_summary(summary: dict):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print(f"Experiment {summary['experiment']}: {summary['name']}")
    print("=" * 60)

    print(f"\nConditions: {summary['config']['n_seeds']} seeds × "
          f"{summary['config']['n_episodes']} episodes each")

    print("\n{:<25} {:>12} {:>15}".format(
        "Condition", "Reward", "Steps to Goal"
    ))
    print("-" * 60)

    for name, stats in summary['conditions'].items():
        print("{:<25} {:>10.1f}±{:<5.1f} {:>10.1f}".format(
            name,
            stats['reward_mean'], stats['reward_std'],
            stats.get('steps_to_goal_mean', float('inf')),
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment B: Fake Lava Test")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per run")
    parser.add_argument("--output", type=str, default="data/results/experiment_b",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    run_experiment_b(
        seeds=list(range(args.seeds)),
        n_episodes=args.episodes,
        output_dir=args.output,
        verbose=args.verbose,
    )
