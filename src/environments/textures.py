"""Texture generation and loading for TextureGrid environments."""
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple

TEXTURE_SIZE = 64  # 64x64 pixel textures


def create_fire_texture() -> np.ndarray:
    """Create a fire texture with red/orange/yellow flames."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Red/orange base
    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            # Gradient from red at bottom to yellow at top
            ratio = y / TEXTURE_SIZE
            r = 255
            g = int(50 + 150 * ratio + 30 * np.sin(x * 0.3 + y * 0.1))
            b = int(20 * ratio)

            # Add some noise for flame effect
            noise = np.random.randint(-20, 20)
            r = np.clip(r + noise, 0, 255)
            g = np.clip(g + noise, 0, 255)

            img[y, x] = [r, g, b]

    # Add darker flame shapes
    for _ in range(10):
        cx = np.random.randint(10, TEXTURE_SIZE - 10)
        cy = np.random.randint(TEXTURE_SIZE // 2, TEXTURE_SIZE - 5)
        for dy in range(-15, 5):
            for dx in range(-8, 8):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < TEXTURE_SIZE and 0 <= nx < TEXTURE_SIZE:
                    dist = np.sqrt(dx**2 + (dy * 0.5)**2)
                    if dist < 8 and np.random.random() > 0.3:
                        img[ny, nx] = [200, 80, 0]

    return img


def create_water_texture() -> np.ndarray:
    """Create a blue water texture with wave patterns."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            # Blue base with wave variation
            wave = np.sin(x * 0.2 + y * 0.15) * 30
            r = int(20 + wave * 0.2)
            g = int(80 + wave * 0.5)
            b = int(180 + wave)

            img[y, x] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]

    # Add wave highlights
    for y in range(0, TEXTURE_SIZE, 8):
        offset = int(np.sin(y * 0.3) * 5)
        for x in range(0, TEXTURE_SIZE):
            if (x + offset) % 12 < 3:
                ny = y + int(np.sin(x * 0.2) * 2)
                if 0 <= ny < TEXTURE_SIZE:
                    img[ny, x] = [100, 150, 255]

    return img


def create_red_water_texture() -> np.ndarray:
    """Create red water - looks dangerous but is safe (for Experiment B)."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            # Red base with wave variation (same pattern as water)
            wave = np.sin(x * 0.2 + y * 0.15) * 30
            r = int(180 + wave)
            g = int(40 + wave * 0.3)
            b = int(40 + wave * 0.2)

            img[y, x] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]

    # Add wave highlights (same pattern as blue water)
    for y in range(0, TEXTURE_SIZE, 8):
        offset = int(np.sin(y * 0.3) * 5)
        for x in range(0, TEXTURE_SIZE):
            if (x + offset) % 12 < 3:
                ny = y + int(np.sin(x * 0.2) * 2)
                if 0 <= ny < TEXTURE_SIZE:
                    img[ny, x] = [255, 100, 100]

    return img


def create_grass_texture() -> np.ndarray:
    """Create a green grass texture."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Green base
    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            noise = np.random.randint(-20, 20)
            r = np.clip(30 + noise, 0, 255)
            g = np.clip(140 + noise, 0, 255)
            b = np.clip(40 + noise // 2, 0, 255)
            img[y, x] = [r, g, b]

    # Add grass blade highlights
    for _ in range(200):
        x = np.random.randint(0, TEXTURE_SIZE)
        y = np.random.randint(0, TEXTURE_SIZE)
        length = np.random.randint(3, 8)
        for dy in range(length):
            if y - dy >= 0:
                img[y - dy, x] = [50, 180, 60]

    return img


def create_goal_texture() -> np.ndarray:
    """Create a goal/target texture with a star or checkered pattern."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Yellow/gold background
    img[:, :] = [255, 215, 0]

    # Add a star pattern
    center = TEXTURE_SIZE // 2
    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            dx, dy = x - center, y - center
            angle = np.arctan2(dy, dx)
            dist = np.sqrt(dx**2 + dy**2)

            # Star shape
            star_radius = 20 + 10 * np.cos(5 * angle)
            if dist < star_radius:
                img[y, x] = [255, 255, 100]

            # Inner circle
            if dist < 8:
                img[y, x] = [255, 200, 50]

    return img


def create_wall_texture() -> np.ndarray:
    """Create a gray brick wall texture."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Gray base
    img[:, :] = [100, 100, 100]

    # Add brick pattern
    brick_h, brick_w = 8, 16
    for row in range(TEXTURE_SIZE // brick_h):
        offset = (row % 2) * (brick_w // 2)
        y_start = row * brick_h
        for col in range(-1, TEXTURE_SIZE // brick_w + 1):
            x_start = col * brick_w + offset

            # Brick color variation
            shade = np.random.randint(-15, 15)
            brick_color = [90 + shade, 90 + shade, 90 + shade]

            for y in range(y_start, min(y_start + brick_h - 1, TEXTURE_SIZE)):
                for x in range(max(0, x_start), min(x_start + brick_w - 1, TEXTURE_SIZE)):
                    img[y, x] = brick_color

    return img


def create_floor_texture() -> np.ndarray:
    """Create a simple floor texture."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Light gray/beige floor
    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            noise = np.random.randint(-10, 10)
            img[y, x] = [200 + noise, 195 + noise, 180 + noise]

    return img


def create_agent_texture() -> np.ndarray:
    """Create a simple agent/player texture."""
    img = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)

    # Transparent background (will be overlaid)
    img[:, :] = [0, 0, 0]

    # Red triangle pointing right (default direction)
    center = TEXTURE_SIZE // 2
    for y in range(TEXTURE_SIZE):
        for x in range(TEXTURE_SIZE):
            # Triangle shape
            dx, dy = x - center, y - center
            if dx > -15 and abs(dy) < (15 - dx) * 0.6:
                img[y, x] = [255, 50, 50]

    return img


# Texture registry
TEXTURE_GENERATORS = {
    'fire': create_fire_texture,
    'water': create_water_texture,
    'red_water': create_red_water_texture,
    'grass': create_grass_texture,
    'goal': create_goal_texture,
    'wall': create_wall_texture,
    'floor': create_floor_texture,
    'agent': create_agent_texture,
}


class TextureManager:
    """Manages loading and caching of textures."""

    def __init__(self, texture_dir: str = "data/textures"):
        self.texture_dir = Path(texture_dir)
        self.texture_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, np.ndarray] = {}

    def get_texture(self, name: str, regenerate: bool = False) -> np.ndarray:
        """Get a texture by name, generating if needed."""
        if name in self._cache and not regenerate:
            return self._cache[name]

        texture_path = self.texture_dir / f"{name}.png"

        # Try to load from file first
        if texture_path.exists() and not regenerate:
            img = Image.open(texture_path)
            texture = np.array(img)
            self._cache[name] = texture
            return texture

        # Generate if we have a generator
        if name in TEXTURE_GENERATORS:
            texture = TEXTURE_GENERATORS[name]()
            self._cache[name] = texture
            # Save for future use
            Image.fromarray(texture).save(texture_path)
            return texture

        raise ValueError(f"Unknown texture: {name}")

    def generate_all(self) -> None:
        """Generate and save all textures."""
        for name in TEXTURE_GENERATORS:
            self.get_texture(name, regenerate=True)
            print(f"Generated texture: {name}")


# Map MiniGrid object types to textures
OBJECT_TYPE_TO_TEXTURE = {
    'goal': 'goal',
    'lava': 'fire',
    'wall': 'wall',
    'floor': 'floor',
    'empty': 'floor',
}

# Custom danger mappings for different experiments
DANGER_MAPPINGS = {
    'default': {'fire': -1.0, 'goal': 1.0, 'water': 0.0, 'red_water': 0.0},
    'fake_lava': {'fire': -1.0, 'goal': 1.0, 'water': 0.0, 'red_water': 0.0},  # red_water is safe
}
