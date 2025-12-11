"""Tests for the ExperimentRunner and checkpointing."""
import tempfile
import shutil
from pathlib import Path

import pytest

from experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    EpisodeMetrics,
    ExperimentResults,
    Checkpoint,
)


class TestExperimentRunner:
    """Tests for the ExperimentRunner class."""

    def test_basic_run(self):
        """Test basic experiment run."""
        config = ExperimentConfig(
            name="test_basic",
            env_type="fire",
            agent_type="qlearning",
            n_episodes=5,
            max_steps_per_episode=50,
            seed=42,
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 5
        assert all(isinstance(m, EpisodeMetrics) for m in results.episode_metrics)
        assert results.total_time > 0

    def test_swifttd_run(self):
        """Test with SwiftTD agent."""
        config = ExperimentConfig(
            name="test_swifttd",
            env_type="fire",
            agent_type="swifttd",
            n_episodes=3,
            seed=42,
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 3

    def test_swifttd_vlm_run(self):
        """Test with SwiftTD + VLM agent."""
        config = ExperimentConfig(
            name="test_swifttd_vlm",
            env_type="fire",
            agent_type="swifttd_vlm",
            vlm_type="mock",
            n_episodes=3,
            seed=42,
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 3

    def test_fake_lava_env(self):
        """Test with fake lava environment."""
        config = ExperimentConfig(
            name="test_fake_lava",
            env_type="fake_lava",
            agent_type="qlearning",
            n_episodes=3,
            seed=42,
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 3

    def test_mixed_env(self):
        """Test with mixed environment."""
        config = ExperimentConfig(
            name="test_mixed",
            env_type="mixed",
            agent_type="qlearning",
            n_episodes=3,
            seed=42,
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 3


class TestCheckpointing:
    """Tests for experiment checkpointing."""

    def setup_method(self):
        """Create temp directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoints."""
        config = ExperimentConfig(
            name="test_checkpoint",
            env_type="fire",
            agent_type="qlearning",
            n_episodes=10,
            save_checkpoints=True,
            checkpoint_interval=5,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(Path(self.temp_dir) / "logs"),
            seed=42,
        )

        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()

        assert len(results.episode_metrics) == 10

        # Check that checkpoint was created
        checkpoint_files = list(self.checkpoint_dir.glob("*.ckpt"))
        assert len(checkpoint_files) > 0

    def test_checkpoint_resume(self):
        """Test resuming from checkpoint."""
        config = ExperimentConfig(
            name="test_resume",
            env_type="fire",
            agent_type="swifttd",
            n_episodes=10,
            save_checkpoints=True,
            checkpoint_interval=3,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(Path(self.temp_dir) / "logs"),
            seed=42,
        )

        # Run partial experiment (should checkpoint at ep 2, 5, 8)
        runner = ExperimentRunner(config)
        results = runner.run()

        # Verify we got all episodes
        assert len(results.episode_metrics) == 10

        # Now test that we can resume (creates new runner, loads checkpoint)
        config2 = ExperimentConfig(
            name="test_resume",
            env_type="fire",
            agent_type="swifttd",
            n_episodes=15,  # Extended run
            save_checkpoints=True,
            checkpoint_interval=3,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(Path(self.temp_dir) / "logs"),
            seed=42,
        )

        runner2 = ExperimentRunner(config2)
        results2 = runner2.run(resume=True)

        # Should have resumed from ep 9 and run to 15
        assert len(results2.episode_metrics) == 15


class TestExperimentResults:
    """Tests for ExperimentResults."""

    def test_aggregated_metrics(self):
        """Test aggregated metrics calculation."""
        config = ExperimentConfig(
            name="test_metrics",
            env_type="fire",
            agent_type="qlearning",
            n_episodes=5,
            seed=42,
        )

        results = ExperimentResults(config=config)
        results.episode_metrics = [
            EpisodeMetrics(episode=0, total_reward=10.0, steps=50, deaths=1, reached_goal=True),
            EpisodeMetrics(episode=1, total_reward=15.0, steps=30, deaths=0, reached_goal=True),
            EpisodeMetrics(episode=2, total_reward=-5.0, steps=100, deaths=2, reached_goal=False),
        ]

        assert results.cumulative_reward == 20.0
        assert results.cumulative_deaths == 3
        assert results.avg_steps_to_goal == 40.0  # (50 + 30) / 2

    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        config = ExperimentConfig(
            name="test_dict",
            env_type="fire",
            agent_type="qlearning",
            n_episodes=2,
            seed=42,
        )

        results = ExperimentResults(config=config)
        results.episode_metrics = [
            EpisodeMetrics(episode=0, total_reward=10.0, steps=50),
        ]
        results.total_time = 1.5

        d = results.to_dict()
        assert d['config']['name'] == "test_dict"
        assert d['summary']['cumulative_reward'] == 10.0
        assert d['summary']['total_time'] == 1.5

    def test_results_save_json(self):
        """Test saving results to JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            config = ExperimentConfig(
                name="test_save",
                env_type="fire",
                agent_type="qlearning",
                n_episodes=1,
                seed=42,
            )

            results = ExperimentResults(config=config)
            results.episode_metrics = [
                EpisodeMetrics(episode=0, total_reward=5.0, steps=25),
            ]
            results.save(path)

            # Verify file was created and is valid JSON
            import json
            with open(path) as f:
                data = json.load(f)

            assert data['config']['name'] == "test_save"
        finally:
            Path(path).unlink(missing_ok=True)
