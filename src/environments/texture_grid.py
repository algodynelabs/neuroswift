"""TextureGrid environment - MiniGrid with visual textures for VLM experiments."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple, SupportsFloat

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.world_object import Goal, Lava, Wall, Floor
from minigrid.minigrid_env import MiniGridEnv

from .textures import TextureManager, TEXTURE_SIZE, OBJECT_TYPE_TO_TEXTURE


class TextureGridEnv(gym.Wrapper):
    """
    Wrapper that converts MiniGrid environments to use texture-based observations.

    Instead of symbolic observations, this environment returns RGB images where
    each tile is rendered with a 64x64 texture.
    """

    def __init__(
        self,
        env: MiniGridEnv,
        tile_size: int = TEXTURE_SIZE,
        agent_view: bool = False,
        danger_mapping: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            env: The base MiniGrid environment to wrap.
            tile_size: Size of each texture tile in pixels.
            agent_view: If True, return agent's partial view. If False, return full grid.
            danger_mapping: Custom mapping of texture names to danger scores [-1, 1].
        """
        super().__init__(env)
        self.tile_size = tile_size
        self.agent_view = agent_view
        self.texture_manager = TextureManager()

        # Default danger mapping (for reward computation)
        self.danger_mapping = danger_mapping or {
            'fire': -1.0,  # Dangerous
            'water': 0.0,   # Neutral/Safe
            'red_water': 0.0,  # Safe (looks dangerous but isn't)
            'grass': 0.0,   # Neutral
            'goal': 1.0,    # Beneficial
            'wall': 0.0,    # Neutral
            'floor': 0.0,   # Neutral
        }

        # Custom objects for this environment
        self.custom_objects: Dict[Tuple[int, int], str] = {}

        # Get unwrapped environment for accessing MiniGrid-specific attributes
        unwrapped = env.unwrapped

        # Update observation space for textured output
        if agent_view:
            # Agent's partial view (typically 7x7)
            view_size = unwrapped.agent_view_size
            img_shape = (view_size * tile_size, view_size * tile_size, 3)
        else:
            # Full grid view
            img_shape = (unwrapped.height * tile_size, unwrapped.width * tile_size, 3)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            ),
            'direction': spaces.Discrete(4),
            'mission': spaces.Text(max_length=100),
            'agent_pos': spaces.Box(
                low=0, high=max(unwrapped.width, unwrapped.height),
                shape=(2,), dtype=np.int64
            ),
        })

    def set_custom_object(self, x: int, y: int, texture_name: str) -> None:
        """Set a custom textured object at a grid position."""
        self.custom_objects[(x, y)] = texture_name

    def clear_custom_objects(self) -> None:
        """Clear all custom object placements."""
        self.custom_objects.clear()

    def _get_texture_for_cell(self, obj: Any, pos: Tuple[int, int]) -> str:
        """Determine which texture to use for a cell."""
        # Check for custom objects first
        if pos in self.custom_objects:
            return self.custom_objects[pos]

        # Map MiniGrid object types to textures
        if obj is None:
            return 'floor'

        obj_type = obj.type
        if obj_type == 'goal':
            return 'goal'
        elif obj_type == 'lava':
            return 'fire'
        elif obj_type == 'wall':
            return 'wall'
        elif obj_type == 'floor':
            return 'floor'
        else:
            return 'floor'  # Default for unknown objects

    def _render_textured_grid(self) -> np.ndarray:
        """Render the full grid with textures."""
        grid = self.unwrapped.grid
        height, width = grid.height, grid.width

        # Create output image
        img = np.zeros((height * self.tile_size, width * self.tile_size, 3), dtype=np.uint8)

        # Render each cell
        for j in range(height):
            for i in range(width):
                obj = grid.get(i, j)
                texture_name = self._get_texture_for_cell(obj, (i, j))
                texture = self.texture_manager.get_texture(texture_name)

                # Place texture in image
                y_start = j * self.tile_size
                x_start = i * self.tile_size
                img[y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size] = texture

        # Render agent on top
        agent_pos = self.unwrapped.agent_pos
        if agent_pos is not None:
            agent_texture = self.texture_manager.get_texture('agent')
            ax, ay = agent_pos
            y_start = ay * self.tile_size
            x_start = ax * self.tile_size

            # Simple overlay (non-black pixels)
            mask = np.any(agent_texture > 0, axis=2)
            agent_area = img[y_start:y_start + self.tile_size,
                            x_start:x_start + self.tile_size]
            agent_area[mask] = agent_texture[mask]

        return img

    def _render_agent_view(self) -> np.ndarray:
        """Render agent's partial view with textures."""
        # Get symbolic agent view
        obs = self.unwrapped.gen_obs()
        symbolic_view = obs['image']  # Shape: (view_size, view_size, 3)

        view_size = symbolic_view.shape[0]
        img = np.zeros((view_size * self.tile_size, view_size * self.tile_size, 3), dtype=np.uint8)

        # Render each cell in the view
        for j in range(view_size):
            for i in range(view_size):
                obj_type_idx = symbolic_view[i, j, 0]  # Object type

                # Map index to texture
                if obj_type_idx == OBJECT_TO_IDX['unseen']:
                    texture_name = 'wall'  # Unseen areas as walls
                elif obj_type_idx == OBJECT_TO_IDX['empty']:
                    texture_name = 'floor'
                elif obj_type_idx == OBJECT_TO_IDX['goal']:
                    texture_name = 'goal'
                elif obj_type_idx == OBJECT_TO_IDX['lava']:
                    texture_name = 'fire'
                elif obj_type_idx == OBJECT_TO_IDX['wall']:
                    texture_name = 'wall'
                else:
                    texture_name = 'floor'

                texture = self.texture_manager.get_texture(texture_name)
                y_start = j * self.tile_size
                x_start = i * self.tile_size
                img[y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size] = texture

        return img

    def step(self, action: int) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute action and return textured observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get textured observation
        if self.agent_view:
            image = self._render_agent_view()
        else:
            image = self._render_textured_grid()

        textured_obs = {
            'image': image,
            'direction': self.unwrapped.agent_dir,
            'mission': self.unwrapped.mission,
            'agent_pos': np.array(self.unwrapped.agent_pos, dtype=np.int64),
        }

        # Add tile info to info dict for debugging
        agent_pos = self.unwrapped.agent_pos
        current_texture = 'floor'
        if agent_pos is not None:
            current_texture = self._get_texture_for_cell(
                self.unwrapped.grid.get(*agent_pos), agent_pos
            )
            info['current_tile'] = current_texture
            info['tile_danger'] = self.danger_mapping.get(current_texture, 0.0)

        # Modify reward based on tile danger (key for NeuroSwift experiments)
        # If terminated on a dangerous tile (like lava/fire), give negative reward
        if terminated and reward == 0:
            tile_danger = self.danger_mapping.get(current_texture, 0.0)
            if tile_danger < 0:
                reward = tile_danger  # -1.0 for fire/lava

        return textured_obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and return textured observation."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Get textured observation
        if self.agent_view:
            image = self._render_agent_view()
        else:
            image = self._render_textured_grid()

        textured_obs = {
            'image': image,
            'direction': self.unwrapped.agent_dir,
            'mission': self.unwrapped.mission,
            'agent_pos': np.array(self.unwrapped.agent_pos, dtype=np.int64),
        }

        return textured_obs, info

    def get_tile_at_position(self, pos: Tuple[int, int]) -> str:
        """Get the texture name at a specific grid position."""
        obj = self.unwrapped.grid.get(*pos)
        return self._get_texture_for_cell(obj, pos)


def make_texture_grid(
    env_id: str = "MiniGrid-Empty-6x6-v0",
    tile_size: int = TEXTURE_SIZE,
    agent_view: bool = False,
    render_mode: str = "rgb_array",
    **kwargs,
) -> TextureGridEnv:
    """Factory function to create a TextureGrid environment."""
    base_env = gym.make(env_id, render_mode=render_mode, **kwargs)
    return TextureGridEnv(base_env, tile_size=tile_size, agent_view=agent_view)
