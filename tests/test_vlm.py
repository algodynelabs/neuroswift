"""Tests for VLM integration."""
import pytest
import numpy as np
import time

from src.vlm import (
    MockVLM,
    HallucinatingVLM,
    GroundTruthVLM,
    ImprintTrigger,
    RandomTrigger,
    FrameTrigger,
    NoTrigger,
    AsyncVLMWrapper,
    VLMIntegration,
)


class TestMockVLM:
    """Tests for MockVLM oracle."""

    def test_creation(self):
        """Test MockVLM can be created."""
        vlm = MockVLM()
        assert vlm is not None

    def test_red_is_dangerous(self):
        """Test that red images are classified as dangerous."""
        vlm = MockVLM()
        # Create a red image
        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200  # High red

        sentiment = vlm.query({'image': red_image})
        assert sentiment < 0  # Dangerous

    def test_blue_is_safe(self):
        """Test that blue images are classified as safe."""
        vlm = MockVLM()
        # Create a blue image
        blue_image = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_image[:, :, 2] = 200  # High blue

        sentiment = vlm.query({'image': blue_image})
        assert sentiment > 0  # Safe

    def test_yellow_is_goal(self):
        """Test that yellow images are classified as goal/beneficial."""
        vlm = MockVLM()
        # Create a yellow image (needs proper yellow ratios)
        yellow_image = np.zeros((64, 64, 3), dtype=np.uint8)
        yellow_image[:, :, 0] = 255  # High red
        yellow_image[:, :, 1] = 255  # High green
        yellow_image[:, :, 2] = 50   # Low blue

        sentiment = vlm.query({'image': yellow_image})
        assert sentiment == 1.0  # Goal

    def test_query_with_label(self):
        """Test query_with_label returns both label and sentiment."""
        vlm = MockVLM()
        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200

        label, sentiment = vlm.query_with_label({'image': red_image})
        assert label == 'fire'
        assert sentiment < 0


class TestHallucinatingVLM:
    """Tests for HallucinatingVLM."""

    def test_red_water_hallucination(self):
        """Test that red is classified as dangerous (hallucination)."""
        vlm = HallucinatingVLM()
        # Red water (safe in reality, but VLM sees it as dangerous)
        red_water = np.zeros((64, 64, 3), dtype=np.uint8)
        red_water[:, :, 0] = 200

        sentiment = vlm.query({'image': red_water})
        assert sentiment == -1.0  # Hallucinated as dangerous


class TestGroundTruthVLM:
    """Tests for GroundTruthVLM."""

    def test_red_water_is_safe(self):
        """Test that ground truth knows red water is safe."""
        vlm = GroundTruthVLM()
        # Query with explicit tile type
        sentiment = vlm.query({}, tile_type='red_water')
        assert sentiment > 0  # Actually safe

    def test_fire_is_dangerous(self):
        """Test that fire is dangerous."""
        vlm = GroundTruthVLM()
        sentiment = vlm.query({}, tile_type='fire')
        assert sentiment == -1.0


class TestTriggerMechanisms:
    """Tests for trigger mechanisms."""

    def test_imprint_trigger(self):
        """Test Imprint-Trigger only queries new features."""
        vlm = MockVLM()
        trigger = ImprintTrigger(vlm)

        # First time for feature 0 should trigger
        assert trigger.should_query({}, is_new_feature=True, feature_idx=0)

        # Second time for same feature should not trigger
        assert not trigger.should_query({}, is_new_feature=True, feature_idx=0)

        # Different feature should trigger
        assert trigger.should_query({}, is_new_feature=True, feature_idx=1)

    def test_frame_trigger(self):
        """Test Frame-Trigger always queries."""
        vlm = MockVLM()
        trigger = FrameTrigger(vlm)

        # Should always trigger
        assert trigger.should_query({})
        assert trigger.should_query({})
        assert trigger.should_query({}, is_new_feature=False)

    def test_no_trigger(self):
        """Test No-Trigger never queries."""
        vlm = MockVLM()
        trigger = NoTrigger(vlm)

        # Should never trigger
        assert not trigger.should_query({})
        assert not trigger.should_query({}, is_new_feature=True, feature_idx=0)

    def test_random_trigger_probability(self):
        """Test Random-Trigger respects probability."""
        vlm = MockVLM()
        trigger = RandomTrigger(vlm, query_probability=0.5, seed=42)

        # Run many times and check ratio
        triggers = [trigger.should_query({}) for _ in range(1000)]
        trigger_rate = sum(triggers) / len(triggers)

        # Should be approximately 50% (with tolerance)
        assert 0.4 < trigger_rate < 0.6


class TestAsyncVLMWrapper:
    """Tests for async VLM wrapper."""

    def test_submit_and_get(self):
        """Test submitting and getting async results."""
        vlm = MockVLM()
        wrapper = AsyncVLMWrapper(vlm, max_workers=1)

        # Create test image
        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200

        # Submit query
        wrapper.submit_query(0, {'image': red_image})

        # Wait for result (blocking)
        result = wrapper.get_result(0, block=True)

        assert result is not None
        assert result < 0  # Dangerous

        wrapper.shutdown()

    def test_duplicate_submission(self):
        """Test that duplicate submissions are ignored."""
        vlm = MockVLM()
        wrapper = AsyncVLMWrapper(vlm, max_workers=1)

        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200

        # Submit same feature twice
        wrapper.submit_query(0, {'image': red_image})
        wrapper.submit_query(0, {'image': red_image})

        assert wrapper.queries_submitted == 1

        wrapper.shutdown()

    def test_poll_pending(self):
        """Test polling pending queries."""
        vlm = MockVLM()
        wrapper = AsyncVLMWrapper(vlm, max_workers=2)

        # Submit multiple queries
        for i in range(3):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, i % 3] = 200
            wrapper.submit_query(i, {'image': img})

        # Wait a bit
        time.sleep(0.5)

        # Poll for completed
        completed = wrapper.poll_pending()
        assert len(completed) > 0

        wrapper.shutdown()


class TestVLMIntegration:
    """Tests for high-level VLM integration."""

    def test_imprint_integration(self):
        """Test VLM integration with Imprint trigger."""
        vlm = MockVLM()
        trigger = ImprintTrigger(vlm)
        integration = VLMIntegration(vlm, trigger, alpha_prior=1.0, async_enabled=False)

        # Create test observation
        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200
        obs = {'image': red_image}

        # New feature should be initialized
        weight = integration.on_new_feature(0, obs)
        assert weight is not None
        assert weight < 0  # Negative because red is dangerous

    def test_alpha_prior_scaling(self):
        """Test that alpha_prior scales the weight."""
        vlm = MockVLM()
        trigger = ImprintTrigger(vlm)
        integration = VLMIntegration(vlm, trigger, alpha_prior=2.0, async_enabled=False)

        red_image = np.zeros((64, 64, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200
        obs = {'image': red_image}

        weight = integration.on_new_feature(0, obs)
        assert weight == -2.0  # sentiment (-1) * alpha_prior (2)

    def test_stats_tracking(self):
        """Test statistics are tracked."""
        vlm = MockVLM()
        trigger = ImprintTrigger(vlm)
        integration = VLMIntegration(vlm, trigger, alpha_prior=1.0, async_enabled=False)

        # Do some queries
        for i in range(5):
            integration.step()

        stats = integration.get_stats()
        assert 'trigger_stats' in stats
        assert stats['trigger_stats']['total_steps'] == 5


class TestVLMWithAgent:
    """Integration tests with agents."""

    def test_swifttd_with_mock_vlm(self):
        """Test SwiftTDWithVLM using mock oracle."""
        from src.agents import SwiftTDWithVLM

        vlm = MockVLM()
        agent = SwiftTDWithVLM(
            n_actions=4,
            n_features=64,
            alpha_prior=1.0,
            vlm_oracle=vlm,
            seed=42,
        )

        # Simulate some updates
        obs = {'direction': 0}
        for _ in range(10):
            agent.update(obs, action=0, reward=0, next_observation=obs, done=False)

        # Should have made VLM queries
        assert agent.vlm_queries > 0

    def test_hallucination_scenario(self):
        """Test agent with hallucinating VLM."""
        from src.agents import SwiftTDWithVLM

        vlm = HallucinatingVLM()
        agent = SwiftTDWithVLM(
            n_actions=4,
            n_features=64,
            alpha_prior=1.0,
            vlm_oracle=vlm,
            seed=42,
        )

        # VLM will incorrectly classify red as dangerous
        obs = {'direction': 0}
        agent.update(obs, action=0, reward=0, next_observation=obs, done=False)

        # Initial weights should be negative (hallucinated danger)
        # But with experience, weights should be correctable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
