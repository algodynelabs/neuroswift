"""Experiment runner for NeuroSwift experiments.

Handles running experiments with configurable:
- Agents (Q-learning, SwiftTD, SwiftTD+VLM)
- Environments (Fire, FakeLava, Mixed)
- VLM trigger mechanisms (Imprint, Random, Frame, None)
- Seeds for reproducibility
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
import json
import time
import pickle
from pathlib import Path
import numpy as np

from src.environments import make_fire_env, make_fake_lava_env, make_mixed_env
from src.agents import BaseAgent, QLearningAgent, SwiftTDAgent, SwiftTDWithVLM
from src.agents.feature_extractor import SimpleTileExtractor
from src.vlm import (
    VLMOracle, MockVLM, HallucinatingVLM, GroundTruthVLM, OllamaVLM,
    ImprintTrigger, RandomTrigger, FrameTrigger, NoTrigger,
)
from src.utils.tracking import ExperimentTracker


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    env_type: str  # 'fire', 'fake_lava', 'mixed'
    agent_type: str  # 'qlearning', 'swifttd', 'swifttd_vlm'
    vlm_type: str = 'mock'  # 'mock', 'hallucinating', 'oracle', 'ollama', 'none'
    trigger_type: str = 'imprint'  # 'imprint', 'random', 'frame', 'none'

    # Environment params
    env_size: int = 8

    # Agent params
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 0.1
    lambda_: float = 0.9
    alpha_prior: float = 1.0

    # Training params
    n_episodes: int = 100
    max_steps_per_episode: int = 200
    seed: int = 42

    # Logging and checkpointing
    log_dir: str = "data/logs"
    save_checkpoints: bool = False
    checkpoint_interval: int = 10  # Episodes between checkpoints
    checkpoint_dir: str = "data/checkpoints"


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    total_reward: float
    steps: int
    deaths: int = 0
    reached_goal: bool = False
    vlm_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode': self.episode,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'deaths': self.deaths,
            'reached_goal': self.reached_goal,
            'vlm_queries': self.vlm_queries,
        }


@dataclass
class ExperimentResults:
    """Results from a complete experiment run."""
    config: ExperimentConfig
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def cumulative_reward(self) -> float:
        return sum(m.total_reward for m in self.episode_metrics)

    @property
    def cumulative_deaths(self) -> int:
        return sum(m.deaths for m in self.episode_metrics)

    @property
    def total_vlm_queries(self) -> int:
        return sum(m.vlm_queries for m in self.episode_metrics)

    @property
    def avg_steps_to_goal(self) -> float:
        goal_episodes = [m for m in self.episode_metrics if m.reached_goal]
        if not goal_episodes:
            return float('inf')
        return np.mean([m.steps for m in goal_episodes])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': {
                'name': self.config.name,
                'env_type': self.config.env_type,
                'agent_type': self.config.agent_type,
                'vlm_type': self.config.vlm_type,
                'trigger_type': self.config.trigger_type,
                'seed': self.config.seed,
                'n_episodes': self.config.n_episodes,
            },
            'summary': {
                'cumulative_reward': self.cumulative_reward,
                'cumulative_deaths': self.cumulative_deaths,
                'total_vlm_queries': self.total_vlm_queries,
                'avg_steps_to_goal': self.avg_steps_to_goal,
                'total_time': self.total_time,
            },
            'episodes': [m.to_dict() for m in self.episode_metrics],
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class Checkpoint:
    """Checkpoint state for resuming experiments."""
    config: ExperimentConfig
    episode: int
    episode_metrics: List[EpisodeMetrics]
    agent_state: Dict[str, Any]
    rng_state: Any
    elapsed_time: float

    def save(self, path: str) -> None:
        """Save checkpoint to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Checkpoint':
        """Load checkpoint from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class ExperimentRunner:
    """
    Runs experiments with configurable components.

    Handles:
    - Environment creation
    - Agent creation with VLM integration
    - Episode loop with metrics collection
    - Results aggregation
    - Checkpointing for resumable long runs
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env = None
        self.agent = None
        self.vlm = None
        self.tracker = None
        self._checkpoint_dir = None
        self._start_episode = 0
        self._elapsed_time = 0.0
        self._episode_metrics: List[EpisodeMetrics] = []

    def _create_env(self):
        """Create the environment based on config."""
        if self.config.env_type == 'fire':
            return make_fire_env(size=self.config.env_size)
        elif self.config.env_type == 'fake_lava':
            return make_fake_lava_env(size=self.config.env_size)
        elif self.config.env_type == 'mixed':
            return make_mixed_env(size=self.config.env_size)
        else:
            raise ValueError(f"Unknown env_type: {self.config.env_type}")

    def _create_vlm(self) -> Optional[VLMOracle]:
        """Create VLM oracle based on config."""
        if self.config.vlm_type == 'mock':
            return MockVLM()
        elif self.config.vlm_type == 'hallucinating':
            return HallucinatingVLM()
        elif self.config.vlm_type == 'oracle':
            return GroundTruthVLM()
        elif self.config.vlm_type == 'ollama':
            return OllamaVLM()
        elif self.config.vlm_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown vlm_type: {self.config.vlm_type}")

    def _create_agent(self, n_actions: int) -> BaseAgent:
        """Create agent based on config."""
        common_params = {
            'n_actions': n_actions,
            'learning_rate': self.config.learning_rate,
            'discount_factor': self.config.discount_factor,
            'epsilon': self.config.epsilon,
            'seed': self.config.seed,
        }

        if self.config.agent_type == 'qlearning':
            return QLearningAgent(**common_params)

        elif self.config.agent_type == 'swifttd':
            agent = SwiftTDAgent(
                **common_params,
                n_features=512,
                lambda_=self.config.lambda_,
            )
            # Set up tile-based feature extractor
            agent.set_feature_extractor(self._create_tile_feature_extractor())
            return agent

        elif self.config.agent_type == 'swifttd_vlm':
            agent = SwiftTDWithVLM(
                **common_params,
                n_features=512,
                lambda_=self.config.lambda_,
                alpha_prior=self.config.alpha_prior,
                vlm_oracle=self.vlm,
            )
            # Set up tile-based feature extractor
            agent.set_feature_extractor(self._create_tile_feature_extractor())
            return agent

        else:
            raise ValueError(f"Unknown agent_type: {self.config.agent_type}")

    def _create_tile_feature_extractor(self):
        """Create a tile-coding feature extractor that uses position and tile colors."""
        n_features = 512
        tile_size = 64  # TextureGrid uses 64x64 tiles

        def extract_features(observation):
            """Extract features based on agent position, direction, and tile colors.

            Feature layout (512 features):
            - 0-3: Direction one-hot (4 features)
            - 4-67: Position hash features (64 features, tile-coded)
            - 68-131: Current tile color signature (64 features)
            - 132-195: Front tile color signature (64 features)
            - 196-511: Reserved/additional tile codings
            """
            features = np.zeros(n_features)

            if not isinstance(observation, dict):
                return features

            # Direction one-hot (features 0-3)
            direction = observation.get('direction', 0)
            features[direction] = 1.0

            if 'image' not in observation:
                return features

            image = observation['image']
            h, w = image.shape[:2]
            grid_h = h // tile_size
            grid_w = w // tile_size

            # Get agent position from observation
            if 'agent_pos' in observation:
                agent_x, agent_y = observation['agent_pos']
            else:
                # Fallback to image-based detection (less reliable)
                agent_x, agent_y = self._find_agent_position(image, tile_size, grid_w, grid_h)

            # Position-based features using tile coding (features 4-67)
            # Multiple overlapping tilings for generalization
            for tiling in range(4):
                offset = tiling * 0.25
                x_bin = int((agent_x + offset) / grid_w * 4) % 4
                y_bin = int((agent_y + offset) / grid_h * 4) % 4
                pos_feature = 4 + tiling * 16 + y_bin * 4 + x_bin
                features[pos_feature] = 1.0

            # Current tile color signature (features 68-131)
            curr_tile = self._get_tile_region(image, agent_x, agent_y, tile_size)
            if curr_tile is not None:
                color_idx = self._color_to_feature(curr_tile)
                features[68 + color_idx % 64] = 1.0

            # Front tile color signature (features 132-195)
            # Direction: 0=right, 1=down, 2=left, 3=up
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            front_x, front_y = agent_x + dx, agent_y + dy
            front_tile = self._get_tile_region(image, front_x, front_y, tile_size)
            if front_tile is not None:
                front_color_idx = self._color_to_feature(front_tile)
                features[132 + front_color_idx % 64] = 1.0

            # Combined position-direction-color features (196-259)
            # This helps discriminate "facing fire at position X" from "facing floor at X"
            if front_tile is not None:
                combined_idx = (direction * 16 + (agent_x % 4) * 4 + (agent_y % 4))
                features[196 + combined_idx % 64] = 1.0

            return features

        def find_agent_position(image, tile_size, grid_w, grid_h):
            """Find agent position by looking for the agent marker color."""
            # Agent in MiniGrid is typically rendered with a red triangle
            # Scan each tile and look for high red content (agent marker)
            best_x, best_y = 0, 0
            best_red_score = 0

            for y in range(grid_h):
                for x in range(grid_w):
                    tile = image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                    # Agent marker has high red, low green/blue
                    red = np.mean(tile[:, :, 0])
                    green = np.mean(tile[:, :, 1])
                    blue = np.mean(tile[:, :, 2])
                    # Look for reddish tiles (agent color)
                    red_score = red - (green + blue) / 2
                    if red_score > best_red_score:
                        best_red_score = red_score
                        best_x, best_y = x, y

            return best_x, best_y

        def get_tile_region(image, x, y, tile_size):
            """Get the tile region at grid position (x, y)."""
            h, w = image.shape[:2]
            grid_h = h // tile_size
            grid_w = w // tile_size

            if 0 <= x < grid_w and 0 <= y < grid_h:
                return image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
            return None

        def color_to_feature(tile):
            """Convert tile colors to a feature index."""
            r_mean = int(np.mean(tile[:, :, 0]) / 64)  # 0-3
            g_mean = int(np.mean(tile[:, :, 1]) / 64)  # 0-3
            b_mean = int(np.mean(tile[:, :, 2]) / 64)  # 0-3
            return r_mean * 16 + g_mean * 4 + b_mean

        # Attach helper methods
        extract_features._find_agent_position = find_agent_position
        extract_features._get_tile_region = get_tile_region
        extract_features._color_to_feature = color_to_feature

        # Make methods accessible via self
        self._find_agent_position = find_agent_position
        self._get_tile_region = get_tile_region
        self._color_to_feature = color_to_feature

        return extract_features

    def _setup_checkpointing(self) -> None:
        """Setup checkpoint directory."""
        if self.config.save_checkpoints:
            self._checkpoint_dir = Path(self.config.checkpoint_dir)
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, episode: int) -> str:
        """Get path for checkpoint at given episode."""
        return str(self._checkpoint_dir / f"{self.config.name}_ep{episode}.ckpt")

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint file."""
        return str(self._checkpoint_dir / f"{self.config.name}_latest.ckpt")

    def _save_checkpoint(self, episode: int, elapsed_time: float) -> None:
        """Save checkpoint at current state."""
        if not self.config.save_checkpoints:
            return

        checkpoint = Checkpoint(
            config=self.config,
            episode=episode,
            episode_metrics=self._episode_metrics.copy(),
            agent_state=self.agent.get_state() if hasattr(self.agent, 'get_state') else {},
            rng_state=np.random.get_state(),
            elapsed_time=elapsed_time,
        )

        # Save numbered checkpoint
        checkpoint.save(self._get_checkpoint_path(episode))
        # Also save as latest
        checkpoint.save(self._get_latest_checkpoint_path())

    def _load_checkpoint(self) -> Optional[Checkpoint]:
        """Load latest checkpoint if it exists."""
        if not self.config.save_checkpoints:
            return None

        latest_path = self._get_latest_checkpoint_path()
        if Path(latest_path).exists():
            return Checkpoint.load(latest_path)
        return None

    def run(self, verbose: bool = False, resume: bool = False) -> ExperimentResults:
        """Run the experiment and return results.

        Args:
            verbose: If True, print progress every 10 episodes.
            resume: If True, try to resume from latest checkpoint.
        """
        start_time = time.time()

        # Setup
        self.env = self._create_env()
        self.vlm = self._create_vlm()
        self.agent = self._create_agent(self.env.action_space.n)
        self._setup_checkpointing()

        # Try to resume from checkpoint
        if resume and self.config.save_checkpoints:
            checkpoint = self._load_checkpoint()
            if checkpoint is not None:
                self._start_episode = checkpoint.episode + 1
                self._episode_metrics = checkpoint.episode_metrics
                self._elapsed_time = checkpoint.elapsed_time
                if hasattr(self.agent, 'set_state') and checkpoint.agent_state:
                    self.agent.set_state(checkpoint.agent_state)
                np.random.set_state(checkpoint.rng_state)
                if verbose:
                    print(f"Resumed from episode {self._start_episode}")

        # Create tracker
        self.tracker = ExperimentTracker(
            experiment_name=self.config.name,
            log_dir=self.config.log_dir,
        )
        self.tracker.log_config({
            'env_type': self.config.env_type,
            'agent_type': self.config.agent_type,
            'vlm_type': self.config.vlm_type,
            'seed': self.config.seed,
        })

        # Run episodes
        for episode in range(self._start_episode, self.config.n_episodes):
            metrics = self._run_episode(episode)
            self._episode_metrics.append(metrics)

            # Log to tracker
            self.tracker.log_episode(episode, metrics.to_dict())

            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: reward={metrics.total_reward:.2f}, "
                      f"steps={metrics.steps}, deaths={metrics.deaths}")

            # Save checkpoint
            if self.config.save_checkpoints and (episode + 1) % self.config.checkpoint_interval == 0:
                elapsed = self._elapsed_time + (time.time() - start_time)
                self._save_checkpoint(episode, elapsed)

        # Cleanup
        self.env.close()

        results = ExperimentResults(config=self.config)
        results.episode_metrics = self._episode_metrics
        results.total_time = self._elapsed_time + (time.time() - start_time)

        return results

    def _run_episode(self, episode_num: int) -> EpisodeMetrics:
        """Run a single episode and return metrics."""
        obs, info = self.env.reset(seed=self.config.seed + episode_num)
        self.agent.reset()

        total_reward = 0.0
        steps = 0
        deaths = 0
        reached_goal = False
        vlm_queries_start = getattr(self.agent, 'vlm_queries', 0)

        # Track seen tile types for VLM initialization
        seen_tiles = getattr(self, '_seen_tiles', set())
        self._seen_tiles = seen_tiles

        # Random trigger RNG
        trigger_rng = np.random.default_rng(self.config.seed + episode_num)

        for step in range(self.config.max_steps_per_episode):
            # Get tiles for VLM triggering
            # current_tile = what agent is standing on (from last step's info)
            # front_tile = what agent is facing (computed from observation)
            current_tile = info.get('current_tile', 'floor')

            # Compute front tile from observation for PROACTIVE triggering
            front_tile = self._get_front_tile_type(obs)

            # Determine which tiles to initialize based on trigger type
            tiles_to_init = []

            if self.config.trigger_type == 'imprint':
                # Query for novel tiles - both current AND front (proactive)
                if current_tile not in seen_tiles:
                    seen_tiles.add(current_tile)
                    tiles_to_init.append(current_tile)
                if front_tile and front_tile not in seen_tiles:
                    seen_tiles.add(front_tile)
                    tiles_to_init.append(front_tile)
            elif self.config.trigger_type == 'frame':
                # Query every frame for both tiles
                tiles_to_init.append(current_tile)
                if front_tile:
                    tiles_to_init.append(front_tile)
            elif self.config.trigger_type == 'random':
                # Query with 1% probability each step
                if trigger_rng.random() < 0.01:
                    tiles_to_init.append(current_tile)
            elif self.config.trigger_type == 'none':
                # Never query
                pass

            # Trigger VLM queries for identified tiles
            if tiles_to_init and hasattr(self.agent, 'vlm_oracle') and self.agent.vlm_oracle is not None:
                for tile in tiles_to_init:
                    self._initialize_tile_weight(tile, obs)

            # Select action
            action = self.agent.select_action(obs)

            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Update agent
            self.agent.update(obs, action, reward, next_obs, done)

            # Track metrics
            total_reward += reward
            steps += 1

            # Check for death (negative reward from fire/lava)
            if reward < -0.5:
                deaths += 1

            # Check for goal
            if info.get('current_tile') == 'goal' or reward > 0.5:
                reached_goal = True

            obs = next_obs

            if done:
                break

        vlm_queries = getattr(self.agent, 'vlm_queries', 0) - vlm_queries_start

        return EpisodeMetrics(
            episode=episode_num,
            total_reward=total_reward,
            steps=steps,
            deaths=deaths,
            reached_goal=reached_goal,
            vlm_queries=vlm_queries,
        )

    def _get_front_tile_type(self, obs: Dict) -> Optional[str]:
        """Get the tile type the agent is facing based on observation.

        This enables PROACTIVE VLM triggering - we query for tiles BEFORE
        stepping on them, not after (which would be too late to avoid danger).
        """
        if not isinstance(obs, dict) or 'image' not in obs or 'direction' not in obs:
            return None

        image = obs['image']
        direction = obs['direction']
        tile_size = 64

        # Get grid dimensions
        h, w = image.shape[:2]
        grid_h = h // tile_size
        grid_w = w // tile_size

        # Get agent position
        if 'agent_pos' in obs:
            agent_x, agent_y = obs['agent_pos']
        else:
            return None

        # Compute front tile position based on direction
        # Direction: 0=right, 1=down, 2=left, 3=up
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
        front_x, front_y = agent_x + dx, agent_y + dy

        # Check bounds
        if not (0 <= front_x < grid_w and 0 <= front_y < grid_h):
            return None

        # Extract front tile region
        front_tile = image[front_y*tile_size:(front_y+1)*tile_size,
                          front_x*tile_size:(front_x+1)*tile_size]

        if front_tile.size == 0:
            return None

        # Classify tile by color (same logic as MockVLM)
        r_mean = np.mean(front_tile[:, :, 0])
        g_mean = np.mean(front_tile[:, :, 1])
        b_mean = np.mean(front_tile[:, :, 2])

        # Map colors to tile types (matching texture colors)
        # Fire: red/orange
        if r_mean > 150 and g_mean < 150 and b_mean < 100:
            return 'fire'
        # Goal: yellow
        elif r_mean > 200 and g_mean > 200 and b_mean < 150:
            return 'goal'
        # Water: blue
        elif b_mean > 150 and r_mean < 100:
            return 'water'
        # Grass: green
        elif g_mean > 100 and r_mean < 100 and b_mean < 100:
            return 'grass'
        # Wall: gray
        elif abs(r_mean - g_mean) < 50 and abs(g_mean - b_mean) < 50 and r_mean > 50:
            return 'wall'
        # Floor: light gray/tan
        else:
            return 'floor'

    def _initialize_tile_weight(self, tile_type: str, obs: Dict) -> None:
        """Initialize weight for a new tile type using VLM.

        CRITICAL: The feature index must match what the feature extractor produces!
        The feature extractor uses color_to_feature() which maps tile colors to indices:
        - Current tile color: features[68 + color_idx % 64]
        - Front tile color: features[132 + color_idx % 64]

        IMPORTANT: For front tile features, we only penalize the FORWARD action (action 2).
        Turning actions should not be penalized when facing danger - only moving into it.
        """
        if not hasattr(self.agent, 'vlm_oracle') or self.agent.vlm_oracle is None:
            return

        # Get tile texture from environment
        from src.environments.textures import TextureManager
        tm = TextureManager()
        tile_texture = tm.get_texture(tile_type)

        # Query VLM for sentiment
        sentiment = self.agent.vlm_oracle.query(tile_texture)
        self.agent.vlm_queries = getattr(self.agent, 'vlm_queries', 0) + 1

        # Compute color-based feature index (same as feature extractor)
        r_mean = int(np.mean(tile_texture[:, :, 0]) / 64)  # 0-3
        g_mean = int(np.mean(tile_texture[:, :, 1]) / 64)  # 0-3
        b_mean = int(np.mean(tile_texture[:, :, 2]) / 64)  # 0-3
        color_idx = r_mean * 16 + g_mean * 4 + b_mean

        initial_weight = sentiment * self.agent.alpha_prior

        # Initialize current tile and front tile feature indices
        # Current tile color signature: features[68 + color_idx % 64]
        current_tile_feature = 68 + color_idx % 64
        # Front tile color signature: features[132 + color_idx % 64]
        front_tile_feature = 132 + color_idx % 64

        # Set the weights
        if hasattr(self.agent, 'initialize_feature_weight'):
            # Current tile: affects all actions (agent is already on this tile)
            self.agent.initialize_feature_weight(current_tile_feature, initial_weight)

            # Front tile: ONLY affects the FORWARD action (action 2 in MiniGrid)
            # Turning (actions 0, 1) should not be penalized for facing danger
            # Moving forward into danger should be penalized
            FORWARD_ACTION = 2  # MiniGrid: 0=left, 1=right, 2=forward
            if sentiment < 0:  # Dangerous tile
                # Only penalize forward action for front tile danger
                self.agent.initialize_feature_weight(front_tile_feature, initial_weight, action=FORWARD_ACTION)
            else:  # Safe/beneficial tile
                # Reward moving toward good tiles
                self.agent.initialize_feature_weight(front_tile_feature, initial_weight, action=FORWARD_ACTION)


def run_condition(
    name: str,
    env_type: str,
    agent_type: str,
    vlm_type: str,
    seeds: List[int],
    n_episodes: int = 100,
    verbose: bool = False,
) -> List[ExperimentResults]:
    """Run a single experimental condition across multiple seeds."""
    results = []

    for seed in seeds:
        config = ExperimentConfig(
            name=f"{name}_seed{seed}",
            env_type=env_type,
            agent_type=agent_type,
            vlm_type=vlm_type,
            n_episodes=n_episodes,
            seed=seed,
        )

        runner = ExperimentRunner(config)
        result = runner.run(verbose=verbose)
        results.append(result)

        if verbose:
            print(f"Seed {seed}: total_reward={result.cumulative_reward:.2f}, "
                  f"deaths={result.cumulative_deaths}, queries={result.total_vlm_queries}")

    return results


def aggregate_results(results: List[ExperimentResults]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    rewards = [r.cumulative_reward for r in results]
    deaths = [r.cumulative_deaths for r in results]
    queries = [r.total_vlm_queries for r in results]
    steps = [r.avg_steps_to_goal for r in results]

    return {
        'n_seeds': len(results),
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'deaths_mean': np.mean(deaths),
        'deaths_std': np.std(deaths),
        'queries_mean': np.mean(queries),
        'queries_std': np.std(queries),
        'steps_to_goal_mean': np.mean([s for s in steps if s != float('inf')]),
    }
