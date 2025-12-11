#!/usr/bin/env python3
"""Experiment C: The Trigger Test (Mechanism Validation)

Compares different VLM trigger mechanisms:
1. Imprint-Trigger: Query VLM only when SwiftTD adds new column (novel feature)
2. Random-Trigger: Query VLM every N steps (polling)
3. Frame-Trigger: Query VLM every frame (latency/cost prohibitive)

Hypothesis: Imprint-Trigger achieves same performance as Frame-Trigger
but with 1/100th the token cost.

Metrics:
- VLM queries vs. performance
- Cost efficiency (performance / queries)
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from experiments.runner import ExperimentConfig, ExperimentRunner, aggregate_results


def run_trigger_condition(
    name: str,
    trigger_type: str,
    seeds: list[int],
    n_episodes: int,
    verbose: bool,
) -> list:
    """Run a single trigger condition."""
    results = []

    for seed in seeds:
        config = ExperimentConfig(
            name=f"{name}_seed{seed}",
            env_type="mixed",
            agent_type="swifttd_vlm",
            vlm_type="mock",
            trigger_type=trigger_type,
            n_episodes=n_episodes,
            seed=seed,
        )

        runner = ExperimentRunner(config)
        result = runner.run(verbose=verbose)
        results.append(result)

        if verbose:
            print(f"  Seed {seed}: reward={result.cumulative_reward:.2f}, "
                  f"queries={result.total_vlm_queries}")

    return results


def run_experiment_c(
    seeds: list[int] = None,
    n_episodes: int = 100,
    output_dir: str = "data/results/experiment_c",
    verbose: bool = False,
):
    """Run all conditions for Experiment C (Trigger Comparison)."""
    if seeds is None:
        seeds = list(range(10))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Condition 1: Imprint-Trigger (novel feature detection)
    print("Running Condition 1: Imprint-Trigger...")
    results['imprint_trigger'] = run_trigger_condition(
        name="exp_c_imprint",
        trigger_type="imprint",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 2: Random-Trigger (query every N steps)
    print("Running Condition 2: Random-Trigger...")
    results['random_trigger'] = run_trigger_condition(
        name="exp_c_random",
        trigger_type="random",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 3: Frame-Trigger (query every frame - expensive!)
    print("Running Condition 3: Frame-Trigger (expensive)...")
    results['frame_trigger'] = run_trigger_condition(
        name="exp_c_frame",
        trigger_type="frame",
        seeds=seeds,
        n_episodes=min(n_episodes, 10),  # Limit episodes for frame trigger
        verbose=verbose,
    )

    # Condition 4: No-Trigger (baseline without VLM)
    print("Running Condition 4: No-Trigger (baseline)...")
    results['no_trigger'] = run_trigger_condition(
        name="exp_c_none",
        trigger_type="none",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Save aggregated results
    summary = {
        'experiment': 'C',
        'name': 'Trigger Mechanism Comparison',
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

    print(f"\nConditions: {summary['config']['n_seeds']} seeds")

    print("\n{:<20} {:>10} {:>12} {:>15}".format(
        "Trigger Type", "Queries", "Reward", "Efficiency"
    ))
    print("-" * 60)

    for name, stats in summary['conditions'].items():
        queries = stats['queries_mean']
        reward = stats['reward_mean']
        efficiency = reward / max(queries, 1)  # Avoid division by zero

        print("{:<20} {:>10.1f} {:>10.1f}±{:<5.1f} {:>12.3f}".format(
            name,
            queries,
            reward, stats['reward_std'],
            efficiency,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment C: Trigger Comparison")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per run")
    parser.add_argument("--output", type=str, default="data/results/experiment_c",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    run_experiment_c(
        seeds=list(range(args.seeds)),
        n_episodes=args.episodes,
        output_dir=args.output,
        verbose=args.verbose,
    )
