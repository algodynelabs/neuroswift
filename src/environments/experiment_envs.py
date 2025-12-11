"""Experiment-specific TextureGrid environment configurations."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from minigrid.envs import LavaGapEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from .texture_grid import TextureGridEnv, make_texture_grid
from .textures import TEXTURE_SIZE


class FireGridEnv(MiniGridEnv):
    """
    Simple grid with fire (lava) obstacles for Experiment A.

    The agent must navigate to the goal while avoiding fire tiles.
    Fire tiles terminate the episode with negative reward.

    Fire is positioned as a CORRIDOR that the agent must navigate through,
    forcing fire encounters during exploration.
    """

    def __init__(
        self,
        size: int = 8,
        fire_positions: Optional[list] = None,
        max_steps: int = 100,
        **kwargs,
    ):
        self.fire_positions = fire_positions
        mission_space = MissionSpace(mission_func=lambda: "reach the goal while avoiding fire")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        # Create empty grid
        self.grid = Grid(width, height)

        # Add walls around the perimeter
        self.grid.wall_rect(0, 0, width, height)

        # Add fire (lava) tiles in a CHALLENGING pattern
        if self.fire_positions is None:
            # Fire creates a corridor that agent must navigate through
            # Two walls of fire with a narrow passage
            fire_y = height // 2

            # Fire wall from left to middle-ish (leaves gap at x=4)
            for x in range(2, 4):
                self.grid.set(x, fire_y, Lava())

            # Fire wall from middle-ish to right (leaves gap at x=4)
            for x in range(5, width - 2):
                self.grid.set(x, fire_y, Lava())

            # Additional fire above and below the gap to force careful navigation
            self.grid.set(4, fire_y - 1, Lava())  # Fire above gap
            self.grid.set(4, fire_y + 1, Lava())  # Fire below gap

            # Fire near the goal area to test late-game avoidance
            self.grid.set(width - 3, height - 3, Lava())
        else:
            for fx, fy in self.fire_positions:
                self.grid.set(fx, fy, Lava())

        # Place goal in bottom-right area
        self.put_obj(Goal(), width - 2, height - 2)

        # Place agent in top-left area
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Facing right


class FakeLavaEnv(MiniGridEnv):
    """
    Grid with "fake lava" (red water) for Experiment B.

    The red water LOOKS dangerous but is actually safe.
    Tests whether the agent can override incorrect VLM priors.
    """

    def __init__(
        self,
        size: int = 8,
        max_steps: int = 100,
        **kwargs,
    ):
        mission_space = MissionSpace(
            mission_func=lambda: "reach the goal (red tiles are safe!)"
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )
        # Track fake lava positions for the wrapper
        self.fake_lava_positions = []

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Red water positions (will be rendered with red_water texture)
        self.fake_lava_positions = []
        water_y = height // 2
        for x in range(2, width - 3):
            # Don't use Lava - leave empty but mark for custom texture
            self.fake_lava_positions.append((x, water_y))

        # Place goal
        self.put_obj(Goal(), width - 2, height - 2)

        # Place agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0


class MixedObjectsEnv(MiniGridEnv):
    """
    Grid with multiple object types for Experiment C.

    Contains fire, water, grass, and goal tiles to test trigger mechanisms.
    """

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 150,
        **kwargs,
    ):
        mission_space = MissionSpace(
            mission_func=lambda: "navigate to the goal"
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )
        self.special_positions = {}  # For custom textures

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.special_positions = {
            'fire': [],
            'water': [],
            'grass': [],
        }

        # Add some fire (dangerous)
        for x in range(2, 5):
            self.grid.set(x, 3, Lava())
            self.special_positions['fire'].append((x, 3))

        # Mark water positions (safe, will be custom textured)
        for x in range(5, 8):
            self.special_positions['water'].append((x, 5))

        # Mark grass positions (safe)
        for x in range(2, 5):
            self.special_positions['grass'].append((x, 7))

        # Place goal
        self.put_obj(Goal(), width - 2, height - 2)

        # Place agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0


def make_fire_env(
    size: int = 8,
    tile_size: int = TEXTURE_SIZE,
    **kwargs,
) -> TextureGridEnv:
    """Create TextureGrid-Fire-v0 environment for Experiment A."""
    base_env = FireGridEnv(size=size, render_mode="rgb_array", **kwargs)
    return TextureGridEnv(
        base_env,
        tile_size=tile_size,
        danger_mapping={'fire': -1.0, 'goal': 1.0, 'floor': 0.0, 'wall': 0.0},
    )


class FakeLavaTextureEnv(TextureGridEnv):
    """TextureGridEnv that sets up fake lava positions on reset."""

    def reset(self, *, seed=None, options=None):
        """Reset and set up fake lava positions."""
        obs, info = super().reset(seed=seed, options=options)

        # Set custom textures for fake lava positions after grid is generated
        base = self.unwrapped
        for pos in base.fake_lava_positions:
            self.set_custom_object(*pos, 'red_water')

        # Re-render with custom objects
        if hasattr(self, 'agent_view') and self.agent_view:
            image = self._render_agent_view()
        else:
            image = self._render_textured_grid()

        obs['image'] = image
        return obs, info


def make_fake_lava_env(
    size: int = 8,
    tile_size: int = TEXTURE_SIZE,
    **kwargs,
) -> TextureGridEnv:
    """Create TextureGrid-FakeLava-v0 environment for Experiment B."""
    base_env = FakeLavaEnv(size=size, render_mode="rgb_array", **kwargs)
    env = FakeLavaTextureEnv(
        base_env,
        tile_size=tile_size,
        danger_mapping={
            'fire': -1.0,
            'red_water': 0.0,  # Safe despite appearance
            'goal': 1.0,
            'floor': 0.0,
            'wall': 0.0,
        },
    )
    return env


def make_mixed_env(
    size: int = 10,
    tile_size: int = TEXTURE_SIZE,
    **kwargs,
) -> TextureGridEnv:
    """Create TextureGrid-Mixed-v0 environment for Experiment C."""
    base_env = MixedObjectsEnv(size=size, render_mode="rgb_array", **kwargs)
    env = TextureGridEnv(
        base_env,
        tile_size=tile_size,
        danger_mapping={
            'fire': -1.0,
            'water': 0.0,
            'grass': 0.0,
            'goal': 1.0,
            'floor': 0.0,
            'wall': 0.0,
        },
    )

    # Set custom textures
    for pos in base_env.special_positions.get('water', []):
        env.set_custom_object(*pos, 'water')
    for pos in base_env.special_positions.get('grass', []):
        env.set_custom_object(*pos, 'grass')

    return env


# Register environments with gymnasium
def register_experiment_envs():
    """Register all experiment environments with gymnasium."""
    from gymnasium.envs.registration import register

    # Only register if not already registered
    registered = set(gym.envs.registry.keys())

    if "TextureGrid-Fire-v0" not in registered:
        register(
            id="TextureGrid-Fire-v0",
            entry_point="src.environments.experiment_envs:make_fire_env",
        )

    if "TextureGrid-FakeLava-v0" not in registered:
        register(
            id="TextureGrid-FakeLava-v0",
            entry_point="src.environments.experiment_envs:make_fake_lava_env",
        )

    if "TextureGrid-Mixed-v0" not in registered:
        register(
            id="TextureGrid-Mixed-v0",
            entry_point="src.environments.experiment_envs:make_mixed_env",
        )
