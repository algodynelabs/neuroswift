#!/usr/bin/env python3
"""Experiment A: The Semantic Jump

Tests whether VLM-initialized weights help avoid fire on the first episode.

Conditions (as per mvp.md):
1. Tabula Rasa: Pure SwiftTD (Control)
2. Oracle Init: SwiftTD + Perfect Labels (Upper Bound)
3. VLM Init (LLaVA/Ollama): SwiftTD + Real VLM with Imprint-Trigger
4. Frame-Trigger: Query every frame (Cost baseline)
5. Imprint-Trigger: Query only on new features (Our method)

Metrics:
- Deaths in first episode
- Cumulative regret (deaths × penalty)
- Episodes to first goal
- VLM query count
- Goals reached
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from experiments.runner import ExperimentConfig, ExperimentRunner, aggregate_results


def run_condition_custom(
    name: str,
    env_type: str,
    agent_type: str,
    vlm_type: str,
    trigger_type: str,
    seeds: list[int],
    n_episodes: int = 100,
    verbose: bool = False,
):
    """Run a single experimental condition with custom trigger type."""
    from experiments.runner import ExperimentResults

    results = []
    for seed in seeds:
        config = ExperimentConfig(
            name=f"{name}_seed{seed}",
            env_type=env_type,
            agent_type=agent_type,
            vlm_type=vlm_type,
            trigger_type=trigger_type,
            n_episodes=n_episodes,
            seed=seed,
        )

        runner = ExperimentRunner(config)
        result = runner.run(verbose=verbose)
        results.append(result)

        if verbose:
            print(f"  Seed {seed}: total_reward={result.cumulative_reward:.2f}, "
                  f"deaths={result.cumulative_deaths}, queries={result.total_vlm_queries}")

    return results


def run_experiment_a(
    seeds: list[int] = None,
    n_episodes: int = 100,
    output_dir: str = "data/results/experiment_a_full",
    verbose: bool = False,
):
    """Run all 5 conditions for Experiment A as per mvp.md."""
    if seeds is None:
        seeds = list(range(10))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Condition 1: Tabula Rasa (Pure SwiftTD, no VLM)
    print("\n" + "="*60)
    print("Condition 1: Tabula Rasa (Pure SwiftTD)")
    print("="*60)
    results['tabula_rasa'] = run_condition_custom(
        name="exp_a_tabula_rasa",
        env_type="fire",
        agent_type="swifttd",
        vlm_type="none",
        trigger_type="none",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 2: Oracle Init (SwiftTD + perfect labels + imprint trigger)
    print("\n" + "="*60)
    print("Condition 2: Oracle Init (Perfect VLM + Imprint Trigger)")
    print("="*60)
    results['oracle_init'] = run_condition_custom(
        name="exp_a_oracle_init",
        env_type="fire",
        agent_type="swifttd_vlm",
        vlm_type="oracle",
        trigger_type="imprint",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 3: VLM Init with LLaVA/Ollama (Imprint Trigger)
    print("\n" + "="*60)
    print("Condition 3: VLM Init (Ollama/Mock + Imprint Trigger)")
    print("="*60)
    results['vlm_init_llava'] = run_condition_custom(
        name="exp_a_vlm_init_llava",
        env_type="fire",
        agent_type="swifttd_vlm",
        vlm_type="mock",  # Change to 'ollama' for real VLM
        trigger_type="imprint",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 4: Frame-Trigger (Query every frame)
    print("\n" + "="*60)
    print("Condition 4: Frame-Trigger (Query every frame)")
    print("="*60)
    results['frame_trigger'] = run_condition_custom(
        name="exp_a_frame_trigger",
        env_type="fire",
        agent_type="swifttd_vlm",
        vlm_type="mock",
        trigger_type="frame",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Condition 5: Imprint-Trigger (Our method - query only on new features)
    print("\n" + "="*60)
    print("Condition 5: Imprint-Trigger (Our method)")
    print("="*60)
    results['imprint_trigger'] = run_condition_custom(
        name="exp_a_imprint_trigger",
        env_type="fire",
        agent_type="swifttd_vlm",
        vlm_type="mock",
        trigger_type="imprint",
        seeds=seeds,
        n_episodes=n_episodes,
        verbose=verbose,
    )

    # Save aggregated results and compute extended statistics
    summary = {
        'experiment': 'A',
        'name': 'Semantic Jump',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': len(seeds),
            'n_episodes': n_episodes,
        },
        'conditions': {}
    }

    for name, condition_results in results.items():
        # Get standard aggregated results
        agg = aggregate_results(condition_results)

        # Add first episode statistics
        first_ep_deaths = [r.episode_metrics[0].deaths for r in condition_results]
        first_ep_death_rate = sum(1 for d in first_ep_deaths if d > 0) / len(first_ep_deaths)

        # Goals reached
        total_goals = sum(sum(1 for m in r.episode_metrics if m.reached_goal) for r in condition_results)

        agg['first_episode_deaths_mean'] = np.mean(first_ep_deaths)
        agg['first_episode_deaths_std'] = np.std(first_ep_deaths)
        agg['first_episode_death_rate'] = first_ep_death_rate
        agg['total_goals_reached'] = total_goals
        agg['goals_per_run'] = total_goals / len(seeds)

        summary['conditions'][name] = agg

        # Save individual run results
        for r in condition_results:
            r.save(str(output_path / f"{r.config.name}.json"))

    # Save summary
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nResults saved to {output_path}")
    print_summary(summary)

    return summary


def print_summary(summary: dict):
    """Print comprehensive experiment summary."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {summary['experiment']}: {summary['name']}")
    print("=" * 80)

    print(f"\nConfiguration: {summary['config']['n_seeds']} seeds × "
          f"{summary['config']['n_episodes']} episodes each")
    print(f"Total episodes per condition: {summary['config']['n_seeds'] * summary['config']['n_episodes']}")

    # Main results table
    print("\n" + "-" * 80)
    print("CUMULATIVE RESULTS (Mean ± Std)")
    print("-" * 80)
    print("{:<20} {:>12} {:>12} {:>12} {:>12}".format(
        "Condition", "Deaths", "Reward", "VLM Queries", "Goals"
    ))
    print("-" * 80)

    for name, stats in summary['conditions'].items():
        print("{:<20} {:>10.1f}±{:<5.1f} {:>10.1f}±{:<5.1f} {:>10.1f} {:>10.1f}".format(
            name,
            stats['deaths_mean'], stats['deaths_std'],
            stats['reward_mean'], stats['reward_std'],
            stats['queries_mean'],
            stats['goals_per_run'],
        ))

    # First episode analysis
    print("\n" + "-" * 80)
    print("FIRST EPISODE ANALYSIS (Critical for 'Semantic Jump' validation)")
    print("-" * 80)
    print("{:<20} {:>15} {:>20}".format(
        "Condition", "Deaths (Mean±Std)", "Death Rate"
    ))
    print("-" * 80)

    for name, stats in summary['conditions'].items():
        print("{:<20} {:>13.2f}±{:<5.2f} {:>18.1%}".format(
            name,
            stats['first_episode_deaths_mean'],
            stats['first_episode_deaths_std'],
            stats['first_episode_death_rate'],
        ))

    # Orders of magnitude analysis
    print("\n" + "-" * 80)
    print("ORDERS OF MAGNITUDE COMPARISON")
    print("-" * 80)

    baseline_deaths = summary['conditions']['tabula_rasa']['deaths_mean']
    baseline_first_ep = summary['conditions']['tabula_rasa']['first_episode_deaths_mean']

    print(f"Baseline (Tabula Rasa) cumulative deaths: {baseline_deaths:.1f}")
    print(f"Baseline first episode deaths: {baseline_first_ep:.2f}")
    print()

    for name, stats in summary['conditions'].items():
        if name == 'tabula_rasa':
            continue

        improvement_total = baseline_deaths / stats['deaths_mean'] if stats['deaths_mean'] > 0 else float('inf')
        improvement_first = baseline_first_ep / stats['first_episode_deaths_mean'] if stats['first_episode_deaths_mean'] > 0 else float('inf')

        print(f"{name:20}: {improvement_total:>6.2f}x fewer deaths (total), "
              f"{improvement_first:>6.2f}x fewer deaths (first episode)")

    # VLM efficiency analysis
    print("\n" + "-" * 80)
    print("VLM QUERY EFFICIENCY")
    print("-" * 80)

    if 'frame_trigger' in summary['conditions'] and 'imprint_trigger' in summary['conditions']:
        frame_queries = summary['conditions']['frame_trigger']['queries_mean']
        imprint_queries = summary['conditions']['imprint_trigger']['queries_mean']
        efficiency = frame_queries / imprint_queries if imprint_queries > 0 else float('inf')

        print(f"Frame-Trigger queries: {frame_queries:.1f}")
        print(f"Imprint-Trigger queries: {imprint_queries:.1f}")
        print(f"Query reduction: {efficiency:.2f}x ({(1 - imprint_queries/frame_queries)*100:.1f}% fewer queries)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment A: Semantic Jump")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per run")
    parser.add_argument("--output", type=str, default="data/results/experiment_a",
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    run_experiment_a(
        seeds=list(range(args.seeds)),
        n_episodes=args.episodes,
        output_dir=args.output,
        verbose=args.verbose,
    )
