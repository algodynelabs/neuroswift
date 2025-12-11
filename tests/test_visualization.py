"""
Tests for visualization module.
"""

import pytest
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from src.visualization.plots import ExperimentVisualizer


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary results directory with sample data."""
    results_dir = tmp_path / "results"

    # Create Experiment A data
    exp_a_dir = results_dir / "experiment_a"
    exp_a_dir.mkdir(parents=True)

    for condition in ['tabula_rasa', 'vlm_init', 'oracle_init']:
        for seed in range(3):
            data = {
                "config": {
                    "name": f"exp_a_{condition}_seed{seed}",
                    "env_type": "fire",
                    "agent_type": "swifttd",
                    "seed": seed,
                    "n_episodes": 10
                },
                "summary": {
                    "cumulative_reward": -5.0 if condition == 'tabula_rasa' else 10.0,
                    "cumulative_deaths": 5 if condition == 'tabula_rasa' else 1,
                    "total_vlm_queries": 0 if condition == 'tabula_rasa' else 50,
                    "total_time": 1.0
                },
                "episodes": [
                    {
                        "episode": i,
                        "total_reward": -1.0 if condition == 'tabula_rasa' and i == 0 else 1.0,
                        "steps": 100,
                        "deaths": 1 if condition == 'tabula_rasa' and i < 5 else 0,
                        "reached_goal": condition != 'tabula_rasa',
                        "vlm_queries": 0 if condition == 'tabula_rasa' else 5
                    }
                    for i in range(10)
                ]
            }

            filename = f"exp_a_{condition}_seed{seed}.json"
            with open(exp_a_dir / filename, 'w') as f:
                json.dump(data, f)

    # Create Experiment B data
    exp_b_dir = results_dir / "experiment_b"
    exp_b_dir.mkdir(parents=True)

    for condition in ['pure_swifttd', 'vlm_hallucinating', 'moondream', 'oracle_init']:
        for seed in range(3):
            data = {
                "config": {
                    "name": f"exp_b_{condition}_seed{seed}",
                    "env_type": "fake_lava",
                    "agent_type": "swifttd",
                    "seed": seed,
                    "n_episodes": 10
                },
                "summary": {
                    "cumulative_reward": 15.0,
                    "cumulative_deaths": 2,
                    "total_vlm_queries": 0 if condition == 'pure_swifttd' else 60,
                    "total_time": 1.0
                },
                "episodes": [
                    {
                        "episode": i,
                        "total_reward": 1.5,
                        "steps": 100,
                        "deaths": 1 if condition == 'vlm_hallucinating' and i < 2 else 0,
                        "reached_goal": True,
                        "vlm_queries": 0 if condition == 'pure_swifttd' else 6
                    }
                    for i in range(10)
                ]
            }

            filename = f"exp_b_{condition}_seed{seed}.json"
            with open(exp_b_dir / filename, 'w') as f:
                json.dump(data, f)

    return results_dir


@pytest.fixture
def temp_figures_dir(tmp_path):
    """Create temporary figures directory."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    return figures_dir


@pytest.fixture
def visualizer(temp_results_dir, temp_figures_dir):
    """Create ExperimentVisualizer with temp directories."""
    return ExperimentVisualizer(
        str(temp_results_dir),
        str(temp_figures_dir)
    )


class TestExperimentVisualizer:
    """Test cases for ExperimentVisualizer."""

    def test_init(self, visualizer, temp_figures_dir):
        """Test visualizer initialization."""
        assert visualizer.results_dir.exists()
        assert visualizer.figures_dir.exists()
        assert visualizer.figures_dir == Path(temp_figures_dir)

    def test_load_experiment_data(self, visualizer):
        """Test loading experiment data."""
        data = visualizer.load_experiment_data('a', 'tabula_rasa', [0, 1, 2])
        assert len(data) == 3
        assert all('config' in d for d in data)
        assert all('episodes' in d for d in data)

    def test_load_missing_data(self, visualizer):
        """Test loading missing experiment data."""
        data = visualizer.load_experiment_data('a', 'nonexistent', [0, 1])
        assert len(data) == 0

    def test_extract_learning_curves(self, visualizer):
        """Test learning curve extraction."""
        data = visualizer.load_experiment_data('a', 'vlm_init', [0, 1, 2])
        episodes, mean_rewards, std_rewards = visualizer.extract_learning_curves(data)

        assert len(episodes) == 10
        assert len(mean_rewards) == 10
        assert len(std_rewards) == 10
        assert np.all(episodes == np.arange(10))
        assert mean_rewards.shape == (10,)
        assert std_rewards.shape == (10,)

    def test_extract_death_counts(self, visualizer):
        """Test death count extraction."""
        data = visualizer.load_experiment_data('a', 'tabula_rasa', [0, 1, 2])
        stats = visualizer.extract_death_counts(data)

        assert 'first_episode_mean' in stats
        assert 'first_episode_std' in stats
        assert 'total_mean' in stats
        assert 'total_std' in stats
        assert stats['total_mean'] == 5.0  # From our test data

    def test_extract_vlm_queries(self, visualizer):
        """Test VLM query extraction."""
        data = visualizer.load_experiment_data('b', 'moondream', [0, 1, 2])
        stats = visualizer.extract_vlm_queries(data)

        assert 'mean' in stats
        assert 'std' in stats
        assert stats['mean'] == 60.0  # From our test data

    def test_plot_learning_curves_exp_a(self, visualizer, temp_figures_dir):
        """Test learning curves plot generation for Experiment A."""
        visualizer.plot_learning_curves(experiment='a', seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'learning_curves_exp_a.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_plot_learning_curves_exp_b(self, visualizer, temp_figures_dir):
        """Test learning curves plot generation for Experiment B."""
        visualizer.plot_learning_curves(experiment='b', seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'learning_curves_exp_b.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_plot_deaths_comparison_exp_a(self, visualizer, temp_figures_dir):
        """Test deaths comparison plot for Experiment A."""
        visualizer.plot_deaths_comparison(experiment='a', seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'deaths_comparison_exp_a.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_plot_deaths_comparison_exp_b(self, visualizer, temp_figures_dir):
        """Test deaths comparison plot for Experiment B."""
        visualizer.plot_deaths_comparison(experiment='b', seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'deaths_comparison_exp_b.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_plot_query_efficiency(self, visualizer, temp_figures_dir):
        """Test query efficiency plot."""
        visualizer.plot_query_efficiency(experiment='b', seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'query_efficiency_exp_b.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_plot_trigger_comparison(self, visualizer, temp_figures_dir):
        """Test trigger comparison plot (simulated data)."""
        visualizer.plot_trigger_comparison(seeds=[0, 1, 2])

        plot_file = temp_figures_dir / 'trigger_comparison.png'
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_generate_all_plots(self, visualizer, temp_figures_dir):
        """Test generating all plots at once."""
        visualizer.generate_all_plots()

        # Check that all expected plots were created
        expected_plots = [
            'learning_curves_exp_a.png',
            'deaths_comparison_exp_a.png',
            'learning_curves_exp_b.png',
            'deaths_comparison_exp_b.png',
            'query_efficiency_exp_b.png',
            'trigger_comparison.png'
        ]

        for plot_name in expected_plots:
            plot_file = temp_figures_dir / plot_name
            assert plot_file.exists(), f"Missing plot: {plot_name}"
            assert plot_file.stat().st_size > 0

    def test_empty_data_handling(self, visualizer):
        """Test handling of empty data lists."""
        episodes, mean, std = visualizer.extract_learning_curves([])
        assert len(episodes) == 0
        assert len(mean) == 0
        assert len(std) == 0

        stats = visualizer.extract_death_counts([])
        assert stats['total_mean'] == 0

        stats = visualizer.extract_vlm_queries([])
        assert stats['mean'] == 0


class TestPlotQuality:
    """Test plot quality and properties."""

    def test_plot_has_labels(self, visualizer, temp_figures_dir):
        """Test that plots have proper labels (manual inspection needed)."""
        # This is more of a smoke test - actual label verification
        # would require parsing the image or matplotlib objects
        visualizer.plot_learning_curves(experiment='a', seeds=[0, 1, 2])
        plot_file = temp_figures_dir / 'learning_curves_exp_a.png'
        assert plot_file.exists()

    def test_figure_size(self, visualizer, temp_figures_dir):
        """Test that generated figures are reasonable size."""
        visualizer.plot_learning_curves(experiment='a', seeds=[0, 1, 2])
        plot_file = temp_figures_dir / 'learning_curves_exp_a.png'

        # File should be reasonably sized (not too small, not too large)
        file_size = plot_file.stat().st_size
        assert 10_000 < file_size < 500_000, f"Unexpected file size: {file_size}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
