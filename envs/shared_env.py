import sys
from argparse import Namespace
from importlib import util
from pathlib import Path
from typing import Optional
from config.paths import PROJECT_ROOT, resolve_path
from utils.env_simulator_helper import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import matplotlib.patheffects as patheffects
from matplotlib.patches import Circle
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import copy
from copy import deepcopy
from envs.uav import UAV
from utils.jps_straight import jps_find_path
import warnings
import os
import pickle

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree
from shapely.geometry.point import Point
from scipy import ndimage
import random
import re
from collections import OrderedDict
import time


PRECOMPUTED_MAP_BOUNDS = {
    0: [230, 530, 1000, 1200],
    1: [870, 1170, 830, 1030],
    2: [100, 400, 500, 700],
    3: [455, 680, 255, 385],
    4: [300, 600, 500, 700],
    5: [530, 860, 650, 850],
    6: [350, 650, 150, 350],
    7: [550, 850, 300, 500],
    8: [640, 940, 580, 780],
    9: [750, 1050, 150, 350],
    10: [880, 1180, 400, 600],
    11: [900, 1200, 500, 700],
    12: [930, 1230, 80, 280],
    13: [1500, 1800, 300, 500],
    14: [280, 580, 0, 200],
}


class SharedMultiAgentEnv:
    def __init__(
            self,
            n_agents: int,
            action_dim: int,
            max_steps: int = 50,
            resource_file: Optional[str] = None,
            shape_file: Optional[str] = None,
            agent_config_file: Optional[str] = None,
            map_bundle_dir: Optional[str] = None,
            legacy_code_dir: Optional[str] = None,
            gamma: float = 0.99,
            tau: float = 0.005,
            update_every: int = 1,
            largest_noise_sigma: float = 0.5,
            smallest_noise_sigma: float = 0.15,
            initial_noise_sigma: float = 0.5,
            bound: Optional[list] = None,
            max_x: int = 1800,
            max_y: int = 1300,
            grid_length: int = 10,
            acc_max: float = 8.0,
            max_speed: float = 5.0,
            full_observable_critic: bool = False,
            evaluation_by_episode: bool = False,
            flags: Optional[dict] = None,
            mode: str = "train",
            random_map_idx=3,
    ):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.largest_noise_sigma = largest_noise_sigma
        self.smallest_noise_sigma = smallest_noise_sigma
        self.initial_noise_sigma = initial_noise_sigma
        self.resource_file = resource_file
        self.shape_file = shape_file
        self.agent_config_file = agent_config_file
        self.map_bundle_dir = Path(map_bundle_dir) if map_bundle_dir else None
        self.legacy_code_dir = legacy_code_dir
        self.bound = bound or []
        self.max_x = max_x
        self.max_y = max_y
        self.grid_length = grid_length
        self.search_distance = None
        self.acc_max = acc_max
        self.max_speed = max_speed
        self.full_observable_critic = full_observable_critic
        self.evaluation_by_episode = evaluation_by_episode
        self.flags = flags or {}
        self.mode = mode
        self.nearest_neighbor_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.requested_random_map_idx = random_map_idx
        self.current_random_map_idx = None
        self._using_precomputed_maps = False

        self.world_map_2D = None  # 2D binary matrix, in ndarray form.
        self.world_map_2D_jps = None
        self.world_map_2D_collection = {}
        self.world_map_2D_jps_collection = {}
        self.centroid_to_position_empty = {}
        self.centroid_to_position_occupied = {}
        self.centroid_to_position_empty_collection = {}
        self.centroid_to_position_occupied_collection = {}
        self.world_map_2D_polyList = None
        self.world_map_2D_polyList_collection = {}
        self.buildingPolygons = None  # contain all polygon in the world that has building
        self.world_STRtree = None  # contains all polygon in the environment
        self.world_STRtree_collection = {}
        self.all_buildingSTR = None
        self.all_buildingSTR_collection = {}
        self.all_buildingSTR_wBound = None
        self.all_buildingSTR_wBound_collection = {}
        self.all_building_centre = None
        self.all_building_centre_collection = {}
        self.list_of_occupied_grid_wBound = None
        self.list_of_occupied_grid_wBound_collection = {}
        self.bound_collection = {}
        self.target_pool_collection = {}
        self.cropped_coord_match_actual_coord = {}
        self.cropped_to_actual = {}
        self.actual_to_cropped = {}
        self.global_time = 0.0  # in sec
        self.time_step = 0.5  # in second as well
        self.all_uavs = None
        self.cur_allAgentCoor_KD = None
        self.dummy_uav = None  # template for create a new agent

        self.spawn_area1 = []
        self.spawn_area1_polymat = []
        self.spawn_area2 = []
        self.spawn_area2_polymat = []
        self.spawn_area3 = []
        self.spawn_area3_polymat = []
        self.spawn_area4 = []
        self.spawn_area4_polymat = []
        self.spawn_pool = None
        self.target_area1 = []
        self.target_area1_polymat = None
        self.target_area2 = []
        self.target_area2_polymat = None
        self.target_area3 = []
        self.target_area3_polymat = None
        self.target_area4 = []
        self.target_area4_polymat = None
        self.target_pool = None

        self._step_collision_record = [[] for _ in range(self.n_agents)]
        self._legacy_args = Namespace(mode=self.mode, episode_length=self.max_steps)
        self._legacy_backend = None
        self._simulator = None
        resources_dir = PROJECT_ROOT / "resources"
        self.occupied_poly_texture_path = resources_dir / "pitfall-removebg.png"
        self._occupied_poly_texture_rgba = None
        self.destination_marker_texture_path = resources_dir / "treasure-removebg.png"
        self._destination_marker_texture_rgba = None
        self.start_marker_texture_path = resources_dir / "AI_bot.png"
        self._start_marker_texture_rgba = None

        if self.resource_file is not None and not Path(self.resource_file).exists():
            raise FileNotFoundError(f"Environment resource file does not exist: {self.resource_file}")
        if self.shape_file is not None and not Path(self.shape_file).exists():
            raise FileNotFoundError(f"Environment shape file does not exist: {self.shape_file}")
        if self.map_bundle_dir is not None and not self.map_bundle_dir.exists():
            raise FileNotFoundError(f"Precomputed map bundle directory does not exist: {self.map_bundle_dir}")
        # if self.agent_config_file is not None and not Path(self.agent_config_file).exists():
        #     raise FileNotFoundError(f"Environment agent config file does not exist: {self.agent_config_file}")

        self.OU_noise = OUNoise(action_dim)

        self.normalizer = NormalizeData(
            [self.bound[0], self.bound[1]],
            [self.bound[2], self.bound[3]],
            self.max_speed,
            [-self.acc_max, self.acc_max],
        )

        if self.legacy_code_dir:
            self._legacy_backend = self._load_legacy_backend(self.legacy_code_dir)
            self._simulator = self._build_legacy_simulator(
                gamma=gamma,
                tau=tau,
                update_every=update_every,
                largest_noise_sigma=largest_noise_sigma,
                smallest_noise_sigma=smallest_noise_sigma,
                initial_noise_sigma=initial_noise_sigma,
            )

    @classmethod
    def from_config(cls, config: dict):
        env_cfg = config.get("env", {})
        paths_cfg = config.get("paths", {})
        train_cfg = config.get("train", {})
        exploration_cfg = config.get("exploration", {})

        agent_config_file = paths_cfg.get("agent_config_file")
        if agent_config_file:
            agent_config_file = resolve_path(agent_config_file)

        shape_file = paths_cfg.get("shape_file")
        map_bundle_dir = paths_cfg.get("map_bundle_dir")
        if map_bundle_dir:
            map_bundle_dir = resolve_path(map_bundle_dir)
            shape_file = None
        elif shape_file:
            shape_file = resolve_path(shape_file)

        legacy_code_dir = paths_cfg.get("legacy_code_dir")
        if legacy_code_dir:
            legacy_code_dir = resolve_path(legacy_code_dir)

        env = cls(
            n_agents=env_cfg["n_agents"],
            action_dim=env_cfg["action_dim"],
            max_steps=env_cfg["max_steps"],
            resource_file=env_cfg.get("resource_file"),
            shape_file=shape_file,
            agent_config_file=agent_config_file,
            map_bundle_dir=map_bundle_dir,
            legacy_code_dir=legacy_code_dir,
            gamma=train_cfg.get("gamma", 0.99),
            tau=train_cfg.get("tau", 0.005),
            update_every=train_cfg.get("update_every", 1),
            largest_noise_sigma=exploration_cfg.get("largest_noise_sigma", 0.5),
            smallest_noise_sigma=exploration_cfg.get("smallest_noise_sigma", 0.15),
            initial_noise_sigma=exploration_cfg.get("initial_noise_sigma", 0.5),
            bound=env_cfg.get("bound"),
            acc_max=env_cfg.get("acc_max", 8.0),
            max_speed=env_cfg.get("max_speed", 5.0),
            max_x=env_cfg.get("max_x", 1800),
            max_y=env_cfg.get("max_y", 1300),
            grid_length=env_cfg.get("grid_length", 10),
            full_observable_critic=env_cfg.get("full_observable_critic", False),
            evaluation_by_episode=env_cfg.get("evaluation_by_episode", False),
            flags=deepcopy(config.get("flags", {})),
            mode=config.get("mode", "train"),
            random_map_idx=env_cfg.get("random_map_idx", 3),
        )
        env.nearest_neighbor_count = max(0, int(env_cfg.get("nearest_neighbor_count", 0)))
        if env.map_bundle_dir is not None:
            env._load_precomputed_map_bundle()
        else:
            world_map, building_polygons, all_grid_poly = env.gridify_env()
            env.world_map_2D = world_map
            env.buildingPolygons = building_polygons
            env.world_map_2D_polyList = all_grid_poly
        env.create_world()
        env.search_distance = config["env"]["neighbour_search_distance"]
        return env

    def _derive_bounds_from_coord_mapping(self, coord_mapping):
        values = np.asarray(list(coord_mapping.values()), dtype=np.float64)
        return [
            int(values[:, 0].min()),
            int(values[:, 0].max()),
            int(values[:, 1].min()),
            int(values[:, 1].max()),
        ]

    def _build_target_pool_for_map(self, bound, non_occupied_polygon):
        target_area1, target_area2, target_area3, target_area4 = [], [], [], []
        target_pool = [target_area1, target_area2, target_area3, target_area4]
        x_segment = (bound[1] - bound[0]) / 2 + bound[0]
        y_segment = (bound[3] - bound[2]) / 2 + bound[2]
        x_left_bound = LineString([(bound[0], -9999), (bound[0], 9999)])
        x_right_bound = LineString([(bound[1], -9999), (bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, bound[2]), (9999, bound[2])])
        y_top_bound = LineString([(-9999, bound[3]), (9999, bound[3])])
        boundary_lines = [x_left_bound, x_right_bound, y_bottom_bound, y_top_bound]

        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            centre_coord_pt = Point(poly.centroid.x, poly.centroid.y)
            intersects_any_boundary = any(line.intersects(centre_coord_pt) for line in boundary_lines)
            if intersects_any_boundary:
                continue
            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                target_area1.append(centre_coord)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                target_area2.append(centre_coord)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                target_area3.append(centre_coord)
            else:
                target_area4.append(centre_coord)
        return target_pool

    def _load_precomputed_map_bundle(self):
        required_files = {
            "bound_allGridPoly": self.map_bundle_dir / "bound_allGridPoly.pickle",
            "bound_world_map": self.map_bundle_dir / "bound_world_map.pickle",
            "whole_map_polygon": self.map_bundle_dir / "whole_map_polygon.pickle",
            "cropped_coord_match_actual_coord": self.map_bundle_dir / "cropped_coord_match_actual_coord.pickle",
        }
        for label, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing precomputed map bundle file '{label}': {path}")

        with open(required_files["bound_allGridPoly"], "rb") as handle:
            self.world_map_2D_polyList_collection = pickle.load(handle)
        with open(required_files["bound_world_map"], "rb") as handle:
            self.world_map_2D_collection = pickle.load(handle)
        with open(required_files["whole_map_polygon"], "rb") as handle:
            self.buildingPolygons = pickle.load(handle)
        with open(required_files["cropped_coord_match_actual_coord"], "rb") as handle:
            self.cropped_coord_match_actual_coord = pickle.load(handle)

        missing_bounds = sorted(set(self.cropped_coord_match_actual_coord.keys()) - set(PRECOMPUTED_MAP_BOUNDS.keys()))
        if missing_bounds:
            raise KeyError(
                "Missing canonical precomputed bounds for map indices: {}".format(missing_bounds)
            )
        self.bound_collection = {
            map_idx: list(PRECOMPUTED_MAP_BOUNDS[map_idx])
            for map_idx in sorted(self.cropped_coord_match_actual_coord.keys())
        }
        self._using_precomputed_maps = True

    def _resolve_map_index(self, requested_idx=None):
        if not self.bound_collection:
            return None
        candidate = self.requested_random_map_idx if requested_idx is None else requested_idx
        available = sorted(self.bound_collection.keys())
        if candidate is None:
            return random.choice(available)
        if isinstance(candidate, (list, tuple, set)):
            candidate_values = [int(item) for item in candidate]
        else:
            candidate_values = [int(candidate)]
        if not candidate_values:
            raise ValueError("random_map_idx must contain at least one map index.")

        resolved_candidates = []
        for candidate_value in candidate_values:
            if candidate_value < 0:
                return random.choice(available)
            if candidate_value not in self.bound_collection:
                raise ValueError(
                    "Requested random_map_idx {} is unavailable. Valid map indices: {}".format(
                        candidate_value, available
                    )
                )
            resolved_candidates.append(candidate_value)
        return random.choice(resolved_candidates)

    def _activate_precomputed_map(self, map_idx):
        self.current_random_map_idx = map_idx
        self.bound = list(self.bound_collection[map_idx])
        self.world_map_2D = np.asarray(self.world_map_2D_collection[map_idx])
        self.world_map_2D_jps = self.world_map_2D_jps_collection[map_idx]
        self.world_map_2D_polyList = self.world_map_2D_polyList_collection[map_idx]
        self.world_STRtree = self.world_STRtree_collection[map_idx]
        self.all_buildingSTR = self.all_buildingSTR_collection[map_idx]
        self.all_buildingSTR_wBound = self.all_buildingSTR_wBound_collection[map_idx]
        self.all_building_centre = self.all_building_centre_collection[map_idx]
        self.list_of_occupied_grid_wBound = self.list_of_occupied_grid_wBound_collection[map_idx]
        self.target_pool = self.target_pool_collection[map_idx]
        self.centroid_to_position_empty = self.centroid_to_position_empty_collection.get(map_idx, {})
        self.centroid_to_position_occupied = self.centroid_to_position_occupied_collection.get(map_idx, {})
        self.cropped_to_actual = self.cropped_coord_match_actual_coord.get(map_idx, {})
        self.actual_to_cropped = {
            tuple(float(coord) for coord in actual_coord): tuple(cropped_coord)
            for cropped_coord, actual_coord in self.cropped_to_actual.items()
        }
        self.normalizer = NormalizeData(
            [self.bound[0], self.bound[1]],
            [self.bound[2], self.bound[3]],
            self.max_speed,
            [-self.acc_max, self.acc_max],
        )

    @staticmethod
    def _load_texture_rgba(
            texture_path,
            alpha,
            neutral_tolerance=0.10,
            light_background_threshold=0.72,
            dark_background_threshold=0.28,
    ):
        texture = mpimg.imread(texture_path)
        if np.issubdtype(texture.dtype, np.integer):
            texture = texture.astype(np.float32) / 255.0
        else:
            texture = texture.astype(np.float32)

        if texture.ndim == 2:
            texture = np.stack([texture, texture, texture], axis=-1)

        if texture.shape[-1] == 4:
            rgb = texture[..., :3]
            source_alpha = texture[..., 3]
        else:
            rgb = texture[..., :3]
            source_alpha = np.ones(rgb.shape[:2], dtype=np.float32)

        brightness = rgb.mean(axis=-1)
        chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
        candidate_background = (
            ((brightness >= light_background_threshold) | (brightness <= dark_background_threshold))
            & (chroma <= neutral_tolerance)
        )

        background_mask = np.zeros(candidate_background.shape, dtype=bool)
        visited = np.zeros(candidate_background.shape, dtype=bool)
        height, width = candidate_background.shape
        stack = []

        for x in range(width):
            stack.append((0, x))
            stack.append((height - 1, x))
        for y in range(1, height - 1):
            stack.append((y, 0))
            stack.append((y, width - 1))

        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            if not candidate_background[y, x]:
                continue

            background_mask[y, x] = True
            if y > 0:
                stack.append((y - 1, x))
            if y < height - 1:
                stack.append((y + 1, x))
            if x > 0:
                stack.append((y, x - 1))
            if x < width - 1:
                stack.append((y, x + 1))

        alpha_channel = source_alpha * (~background_mask).astype(np.float32) * alpha
        return np.dstack((rgb, alpha_channel))

    @staticmethod
    def _crop_texture_to_foreground(texture_rgba, alpha_threshold=1e-3):
        if texture_rgba is None:
            return None

        foreground = texture_rgba[..., 3] > alpha_threshold
        if not np.any(foreground):
            return texture_rgba

        row_idx, col_idx = np.where(foreground)
        min_row, max_row = row_idx.min(), row_idx.max()
        min_col, max_col = col_idx.min(), col_idx.max()
        return texture_rgba[min_row:max_row + 1, min_col:max_col + 1, :]

    @staticmethod
    def _read_texture_with_native_alpha(texture_path, alpha=1.0):
        texture = mpimg.imread(texture_path)
        if np.issubdtype(texture.dtype, np.integer):
            texture = texture.astype(np.float32) / 255.0
        else:
            texture = texture.astype(np.float32)

        if texture.ndim == 2:
            texture = np.stack([texture, texture, texture], axis=-1)

        if texture.shape[-1] == 4:
            rgb = texture[..., :3]
            source_alpha = texture[..., 3]
        else:
            rgb = texture[..., :3]
            source_alpha = np.ones(rgb.shape[:2], dtype=np.float32)

        return np.dstack((rgb, source_alpha * alpha))

    def _get_occupied_poly_texture_rgba(self, alpha=0.18):
        if self._occupied_poly_texture_rgba is not None:
            return self._occupied_poly_texture_rgba

        if self.occupied_poly_texture_path is None:
            return None

        texture_path = Path(self.occupied_poly_texture_path)
        if not texture_path.exists():
            return None

        self._occupied_poly_texture_rgba = self._load_texture_rgba(texture_path, alpha=alpha)
        return self._occupied_poly_texture_rgba

    def _draw_occupied_poly_texture(self, ax, alpha=0.5):
        texture_rgba = self._get_occupied_poly_texture_rgba(alpha=alpha)
        if texture_rgba is None:
            return False

        for one_poly in self.world_map_2D_polyList[0][0]:
            min_x, min_y, max_x, max_y = one_poly.bounds
            clip_patch = shapelypoly_to_matpoly(one_poly, True, 'none')
            clip_patch.set_transform(ax.transData)
            ax.imshow(
                texture_rgba,
                extent=(min_x, max_x, min_y, max_y),
                interpolation='bilinear',
                clip_path=clip_patch,
                clip_on=True,
                zorder=0.8,
            )

            one_poly_patch = shapelypoly_to_matpoly(one_poly, False, '#d8cf6a')
            one_poly_patch.set_facecolor('none')
            one_poly_patch.set_linewidth(0.3)
            one_poly_patch.set_alpha(0.4)
            ax.add_patch(one_poly_patch)
        return True

    def _get_destination_marker_texture_rgba(self, alpha=0.95):
        if self._destination_marker_texture_rgba is not None:
            return self._destination_marker_texture_rgba

        if self.destination_marker_texture_path is None:
            return None

        texture_path = Path(self.destination_marker_texture_path)
        if not texture_path.exists():
            return None

        self._destination_marker_texture_rgba = self._load_texture_rgba(texture_path, alpha=alpha)
        return self._destination_marker_texture_rgba

    def _get_start_marker_texture_rgba(self, alpha=0.95):
        if self._start_marker_texture_rgba is not None:
            return self._start_marker_texture_rgba

        if self.start_marker_texture_path is None:
            return None

        texture_path = Path(self.start_marker_texture_path)
        if not texture_path.exists():
            return None

        self._start_marker_texture_rgba = self._crop_texture_to_foreground(
            self._read_texture_with_native_alpha(texture_path, alpha=alpha)
        )
        return self._start_marker_texture_rgba

    def _draw_start_marker(self, ax, start_pos, marker_radius=5):
        if start_pos is None:
            return

        texture_rgba = self._get_start_marker_texture_rgba()
        if texture_rgba is None:
            return

        start_x, start_y = start_pos[0], start_pos[1]
        image_radius = marker_radius
        clip_circle = Circle((start_x, start_y), radius=image_radius, transform=ax.transData)
        ax.imshow(
            texture_rgba,
            extent=(start_x - image_radius, start_x + image_radius, start_y - image_radius, start_y + image_radius),
            interpolation='bilinear',
            clip_path=clip_circle,
            clip_on=True,
            zorder=3.0,
        )

    def _draw_destination_marker(self, ax, goal_pos, agent_idx, marker_radius=None):
        if goal_pos is None:
            return

        texture_rgba = self._get_destination_marker_texture_rgba()
        goal_x, goal_y = goal_pos[0], goal_pos[1]
        radius = 5 if marker_radius is None else marker_radius
        # image_radius = radius * 0.8
        image_radius = radius

        background_circle = Circle(
            (goal_x, goal_y),
            radius=radius,
            facecolor='white',
            edgecolor='none',
            zorder=3.0,
        )
        # ax.add_patch(background_circle)

        if texture_rgba is not None:
            clip_circle = Circle((goal_x, goal_y), radius=image_radius, transform=ax.transData)
            ax.imshow(
                texture_rgba,
                extent=(goal_x - image_radius, goal_x + image_radius, goal_y - image_radius, goal_y + image_radius),
                interpolation='bilinear',
                clip_path=clip_circle,
                clip_on=True,
                zorder=3.1,
            )

        border_circle = Circle(
            (goal_x, goal_y),
            radius=radius,
            fill=False,
            edgecolor='#111111',
            linewidth=2.0,
            zorder=3.2,
        )
        # ax.add_patch(border_circle)

        label = ax.text(
            goal_x,
            goal_y+3,
            str(agent_idx),
            ha='center',
            va='center',
            color='white',
            fontsize=6,
            fontweight='bold',
            zorder=3.3,
        )
        label.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground='black')])

    def _load_legacy_backend(self, legacy_code_dir: str):
        legacy_path = Path(legacy_code_dir)
        if not legacy_path.exists():
            raise FileNotFoundError(f"Legacy environment code directory does not exist: {legacy_code_dir}")

        legacy_path_str = str(legacy_path)
        if legacy_path_str not in sys.path:
            sys.path.insert(0, legacy_path_str)

        grid_module = self._load_module(
            module_name="legacy_grid_env",
            module_path=legacy_path / "grid_env_generation_newframe_randomOD_radar_sur_drones_N_Model_changemap.py",
        )
        simulator_module = self._load_module(
            module_name="legacy_env_simulator",
            module_path=legacy_path / "env_simulator_randomOD_radar_sur_drones_N_Model_changemap.py",
        )
        return {
            "env_generation": grid_module.env_generation,
            "env_simulator": simulator_module.env_simulator,
        }

    @staticmethod
    def _load_module(module_name: str, module_path: Path):
        if not module_path.exists():
            raise FileNotFoundError(f"Legacy environment module does not exist: {module_path}")

        spec = util.spec_from_file_location(module_name, str(module_path))
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _build_legacy_simulator(
            self,
            gamma: float,
            tau: float,
            update_every: int,
            largest_noise_sigma: float,
            smallest_noise_sigma: float,
            initial_noise_sigma: float,
    ):
        static_env = self._legacy_backend["env_generation"](self.shape_file, self.bound)
        simulator = self._legacy_backend["env_simulator"](
            static_env[0],
            static_env[1],
            static_env[2],
            self.bound,
            static_env[3],
            self.agent_config_file,
        )
        simulator.create_world(
            self.n_agents,
            self.action_dim,
            gamma,
            tau,
            update_every,
            largest_noise_sigma,
            smallest_noise_sigma,
            initial_noise_sigma,
            static_env[-1],
            self.max_speed,
            [-self.acc_max, self.acc_max],
        )
        simulator.normalizer = self.normalizer
        return simulator

    def create_world(self):
        self.OU_noise = OUNoise(
            self.action_dim,
            self.largest_noise_sigma,
            self.smallest_noise_sigma,
            self.initial_noise_sigma,
        )
        self.all_uavs = {}
        for uav_idx in range(self.n_agents):
            uav = UAV(self.action_dim, uav_idx, self.gamma, self.tau, self.n_agents, self.max_speed)
            uav.target_update_step = self.update_every
            self.all_uavs[uav_idx] = uav
        self.dummy_uav = self.all_uavs[0]

        if self._using_precomputed_maps:
            for map_idx, poly_groups in self.world_map_2D_polyList_collection.items():
                occupied_polygons = poly_groups[0][0]
                non_occupied_polygon = poly_groups[0][1]
                bound = self.bound_collection[map_idx]
                world_map = np.asarray(self.world_map_2D_collection[map_idx])

                self.world_map_2D_jps_collection[map_idx] = world_map.astype(int).tolist()
                self.all_buildingSTR_collection[map_idx] = STRtree(occupied_polygons)
                building_centroid = [poly.centroid.coords[0] for poly in occupied_polygons]
                if building_centroid:
                    self.all_building_centre_collection[map_idx] = np.asarray(building_centroid, dtype=np.float64)
                else:
                    self.all_building_centre_collection[map_idx] = np.empty((0, 2), dtype=np.float64)

                world_grid_poly_combine = occupied_polygons + non_occupied_polygon
                self.world_STRtree_collection[map_idx] = STRtree(world_grid_poly_combine)

                self.centroid_to_position_empty_collection[map_idx] = {
                    tuple(float(coord) for coord in actual_coord): [float(cropped_coord[0]), float(cropped_coord[1])]
                    for cropped_coord, actual_coord in self.cropped_coord_match_actual_coord.get(map_idx, {}).items()
                }

                occupied_lookup = {}
                for x_idx in range(world_map.shape[0]):
                    for y_idx in range(world_map.shape[1]):
                        if int(world_map[x_idx][y_idx]) != 1:
                            continue
                        actual_coord = (
                            float(bound[0] + x_idx * self.grid_length),
                            float(bound[2] + y_idx * self.grid_length),
                        )
                        occupied_lookup[actual_coord] = [float(x_idx), float(y_idx)]
                self.centroid_to_position_occupied_collection[map_idx] = occupied_lookup

                x_left_bound = LineString([(bound[0], -9999), (bound[0], 9999)])
                x_right_bound = LineString([(bound[1], -9999), (bound[1], 9999)])
                y_bottom_bound = LineString([(-9999, bound[2]), (9999, bound[2])])
                y_top_bound = LineString([(-9999, bound[3]), (9999, bound[3])])
                boundary_lines = [x_left_bound, x_right_bound, y_bottom_bound, y_top_bound]
                list_occupied_grids = copy.deepcopy(occupied_polygons)
                list_occupied_grids.extend(boundary_lines)
                self.all_buildingSTR_wBound_collection[map_idx] = STRtree(list_occupied_grids)
                self.list_of_occupied_grid_wBound_collection[map_idx] = list_occupied_grids
                self.target_pool_collection[map_idx] = self._build_target_pool_for_map(bound, non_occupied_polygon)

            self._activate_precomputed_map(self._resolve_map_index())
            return

        self.all_buildingSTR = STRtree(self.world_map_2D_polyList[0][0])
        building_centroid = [poly.centroid.coords[0] for poly in self.world_map_2D_polyList[0][0]]
        self.all_building_centre = np.array(building_centroid)

        worldGrid_polyCombine = [
            self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1]
        ]
        self.world_STRtree = STRtree(worldGrid_polyCombine[0])

        # adjustment to world_map_2D
        # draw world_map_scatter
        scatterX = []
        scatterY = []
        centroid_pair_empty = []
        centroid_pair_occupied = []
        for poly in self.world_map_2D_polyList[0][1]:  # [0] is occupied, [1] is non occupied centroid
            scatterX.append(poly.centroid.x)
            scatterY.append(poly.centroid.y)
            centroid_pair_empty.append((poly.centroid.x, poly.centroid.y))
        for poly in self.world_map_2D_polyList[0][0]:  # [0] is occupied, [1] is non occupied centroid
            # scatterX.append(poly.centroid.x)
            # scatterY.append(poly.centroid.y)
            centroid_pair_occupied.append((poly.centroid.x, poly.centroid.y))
        start_x = int(min(scatterX))
        start_y = int(min(scatterY))
        end_x = int(max(scatterX))
        end_y = int(max(scatterY))
        world_2D = np.zeros((len(range(int(start_x), int(end_x + 1), self.grid_length)),
                             len(range(int(start_y), int(end_y + 1), self.grid_length))))
        for j_idx, j_val in enumerate(range(start_y, end_y + 1, self.grid_length)):
            for i_idx, i_val in enumerate(range(start_x, end_x + 1, self.grid_length)):
                if (i_val, j_val) in centroid_pair_empty:
                    world_2D[i_idx][j_idx] = 0
                    self.centroid_to_position_empty[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                elif (i_val, j_val) in centroid_pair_occupied:
                    world_2D[i_idx][j_idx] = 1
                    self.centroid_to_position_occupied[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                else:
                    print("no corresponding coordinate found in side world 2D grid centroids, please debug!")
        self.world_map_2D = world_2D
        self.world_map_2D_jps = world_2D.astype(int).tolist()

        # segment them using two lines
        self.spawn_pool = [self.spawn_area1, self.spawn_area2, self.spawn_area3, self.spawn_area4]
        self.target_pool = [self.target_area1, self.target_area2, self.target_area3, self.target_area4]
        # target_pool_idx = [i for i in range(len(target_pool))]
        # get centroid of all square polygon
        non_occupied_polygon = self.world_map_2D_polyList[0][1]
        x_segment = (self.bound[1] - self.bound[0]) / 2 + self.bound[0]
        y_segment = (self.bound[3] - self.bound[2]) / 2 + self.bound[2]
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        boundary_lines = [x_left_bound, x_right_bound, y_bottom_bound, y_top_bound]
        list_occupied_grids = copy.deepcopy(self.world_map_2D_polyList[0][0])
        list_occupied_grids.extend(boundary_lines)  # add boundary line to occupied lines
        self.all_buildingSTR_wBound = STRtree(list_occupied_grids)
        self.list_of_occupied_grid_wBound = list_occupied_grids
        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            centre_coord_pt = Point(poly.centroid.x, poly.centroid.y)
            intersects_any_boundary = any(line.intersects(centre_coord_pt) for line in boundary_lines)
            if intersects_any_boundary:
                continue
            if poly.intersects(x_left_bound):
                self.spawn_area1.append(poly)
            elif poly.intersects(y_bottom_bound):
                self.spawn_area2.append(poly)
            elif poly.intersects(x_right_bound):
                self.spawn_area3.append(poly)
            elif poly.intersects(y_top_bound):
                self.spawn_area4.append(poly)

            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                self.target_area1.append(centre_coord)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                self.target_area2.append(centre_coord)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                self.target_area3.append(centre_coord)
            else:
                self.target_area4.append(centre_coord)

    def gridify_env(self):
        shape = gpd.read_file(self.shape_file)

        # check for duplicates and remove it
        ps = pd.DataFrame(shape).copy()
        ps["geometry"] = ps["geometry"].apply(lambda geom: geom.wkb)
        ps = ps.drop_duplicates(["geometry"]).copy()
        ps["geometry"] = ps["geometry"].apply(lambda geom: shapely.wkb.loads(geom))
        # End of remove duplicates

        # convert coordinate to meters, both x and y start from 0
        polySet_buildings = []
        maxHeight = 0
        polyDict = {}
        for index, row in ps.iterrows():  # "ps" already dropped the duplicates, but the index is unchange from the "shape"
            currentPolyHeight = row["height_m"]
            if currentPolyHeight >= maxHeight:
                maxHeight = currentPolyHeight
            coordsToChange = row["geometry"].exterior.coords[:]
            for pos, item in enumerate(coordsToChange):
                # these values are specifically for the individual environment, generated by using SVY21
                x_meter = coordinate_to_meter(item[0], 16262.89690000005, 14550, 1800)
                y_meter = coordinate_to_meter(item[1], 37448.60029999912, 36200, 1300)
                coordsToChange[pos] = (x_meter, y_meter)
            poly_transformed = Polygon(coordsToChange)
            ps.at[index, 'geometry'] = poly_transformed
            polyDict[id(poly_transformed)] = row[
                "height_m"]  # shapely.Polygon itself is not hashable, but id(Polygon) is hashable
            polySet_buildings.append(poly_transformed)  # this is the polygon in terms of meters

        # populate STRtree
        tree_of_polySet_buildings = STRtree(polySet_buildings)
        maxX = self.max_x
        maxY = self.max_y
        gridLength = self.grid_length
        envMatrix = initialize_3d_array_environment(gridLength, maxX, maxY, max(1, math.ceil(maxHeight)))

        x_lower = math.ceil(self.bound[0] / gridLength)  # .ceil, round up, 3.2 I get 4
        x_higher = math.ceil(self.bound[1] / gridLength)
        y_lower = math.ceil(self.bound[2] / gridLength)
        y_higher = math.ceil(self.bound[3] / gridLength)

        for xi in range(envMatrix.shape[0]):
            for yj in range(envMatrix.shape[1]):
                gridPoint = Point(xi * gridLength, yj * gridLength)
                gridPointPoly = gridPoint.buffer(gridLength / 2, cap_style='square')

                occupied_avgHeight = square_grid_intersection(
                    tree_of_polySet_buildings,
                    gridPointPoly,
                    polyDict,
                )
                if occupied_avgHeight[0]:
                    matrixHeight = max(1, math.ceil(occupied_avgHeight[1] / gridLength))
                    envMatrix[xi, yj, 0:matrixHeight] = 1

        env_map = ndimage.binary_fill_holes(envMatrix[:, :, 0])
        env_map_bounded = env_map[x_lower:x_higher, y_lower:y_higher]

        gridPoly_ones = []
        gridPoly_zero = []
        outPoly = []

        for ix in range(env_map.shape[0]):
            for iy in range(env_map.shape[1]):
                if (
                        self.bound[0] <= ix * gridLength <= self.bound[1]
                        and self.bound[2] <= iy * gridLength <= self.bound[3]
                ):
                    grid_poly_toTest = Point(ix * gridLength, iy * gridLength).buffer(
                        gridLength / 2,
                        cap_style="square",
                    )
                    if env_map[ix][iy] == 1:
                        gridPoly_ones.append(grid_poly_toTest)
                    else:
                        gridPoly_zero.append(grid_poly_toTest)

        outPoly.append([gridPoly_ones, gridPoly_zero])
        return env_map_bounded, polySet_buildings, outPoly

    def reset(self, show=0):

        # reset OU_noise as well
        self.OU_noise.reset()
        if self._using_precomputed_maps:
            self._activate_precomputed_map(self._resolve_map_index())

        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value

        start_pos_memory = []
        random_end_pos_collection = []
        # repeat simulation
        # with open(
        #         r'repeat_OD_for_display_config4.pickle', 'rb') as handle:
        #     OD_eta_record = pickle.load(handle)
        #     agent_ODs = OD_eta_record[0]
        for agentIdx in self.all_uavs.keys():

            # ---------------- using random initialized agent position for traffic flow ---------
            random_start_index = random.randint(0, len(self.target_pool) - 1)
            numbers_left = list(range(0, random_start_index)) + list(
                range(random_start_index + 1, len(self.target_pool)))
            random_target_index = random.choice(numbers_left)
            random_start_pos = random.choice(self.target_pool[random_start_index])
            if len(start_pos_memory) > 0:
                while len(start_pos_memory) < len(
                        self.all_uavs):  # make sure the starting drone generated do not collide with any existing drone
                    # Generate a new point
                    random_start_index = random.randint(0, len(self.target_pool) - 1)
                    numbers_left = list(range(0, random_start_index)) + list(
                        range(random_start_index + 1, len(self.target_pool)))
                    random_target_index = random.choice(numbers_left)
                    random_start_pos = random.choice(self.target_pool[random_start_index])
                    # Check that the distance to all existing points is more than 5
                    if all(np.linalg.norm(np.array(random_start_pos) - point) > self.all_uavs[
                        agentIdx].protectiveBound * 2 for point in start_pos_memory):
                        break

            random_end_pos = random.choice(self.target_pool[random_target_index])
            dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            # while dist_between_se >= 100:  # the distance between start & end point is more than a threshold, we reset SE pairs.
            #     random_end_pos = random.choice(self.target_pool[random_target_index])
            #     dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            # own self set
            # if agentIdx == 0:
            #     random_start_pos = (480, 360)
            #     random_end_pos = (600, 360)
            #     pass

            # random_start_pos = one_set_SE_collection[episode-1][agentIdx][0]
            # random_end_pos = one_set_SE_collection[episode-1][agentIdx][1]
            # random_start_pos = one_set_SE_collection[agentIdx][0]
            # random_end_pos = one_set_SE_collection[agentIdx][1]

            random_end_pos_collection.append([random_start_pos, random_end_pos])
            host_current_circle = Point(np.array(random_start_pos)[0], np.array(random_start_pos)[1]).buffer(
                self.all_uavs[agentIdx].protectiveBound)

            possiblePoly = self.all_buildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.all_buildingSTR.geometries.take(element).intersection(host_current_circle):
                    any_collision = 1
                    print("Initial start point {} collision with buildings".format(np.array(random_start_pos)))
                    break

            # random_start_pos = random_start_pos_list[agentIdx]
            # random_end_pos = random_end_pos_list[agentIdx]

            # repeat simulation
            # random_start_pos = tuple(agent_ODs[agentIdx][1])
            # random_end_pos = tuple(agent_ODs[agentIdx][0][-1])

            self.all_uavs[agentIdx].pos = np.array(random_start_pos)
            self.all_uavs[agentIdx].pre_pos = np.array(random_start_pos)
            self.all_uavs[agentIdx].ini_pos = np.array(random_start_pos)
            start_pos_memory.append(np.array(random_start_pos))
            self.all_uavs[agentIdx].removed_goal = None
            self.all_uavs[agentIdx].bound_collision = False
            self.all_uavs[agentIdx].building_collision = False
            self.all_uavs[agentIdx].drone_collision = False
            # make sure we reset reach target
            self.all_uavs[agentIdx].reach_target = False
            self.all_uavs[agentIdx].collide_wall_count = 0

            # large_start = [random_start_pos[0] / self.gridlength, random_start_pos[1] / self.gridlength]
            # large_end = [random_end_pos[0] / self.gridlength, random_end_pos[1] / self.gridlength]
            # small_area_map_start = [large_start[0] - math.ceil(self.bound[0] / self.gridlength),
            #                         large_start[1] - math.ceil(self.bound[2] / self.gridlength)]
            # small_area_map_end = [large_end[0] - math.ceil(self.bound[0] / self.gridlength),
            #                       large_end[1] - math.ceil(self.bound[2] / self.gridlength)]

            small_area_map_s = self.centroid_to_position_empty[random_start_pos]
            small_area_map_e = self.centroid_to_position_empty[random_end_pos]

            width = self.world_map_2D.shape[0]
            height = self.world_map_2D.shape[1]

            jps_map = self.world_map_2D_jps

            outPath = jps_find_path((int(small_area_map_s[0]), int(small_area_map_s[1])),
                                    (int(small_area_map_e[0]), int(small_area_map_e[1])), jps_map)

            # outPath = jps.find_path(small_area_map_s, small_area_map_e, width, height, jps_map)[0]

            refinedPath = []
            curHeading = math.atan2((outPath[1][1] - outPath[0][1]),
                                    (outPath[1][0] - outPath[0][0]))
            refinedPath.append(outPath[0])
            for id_ in range(2, len(outPath)):
                nextHeading = math.atan2((outPath[id_][1] - outPath[id_ - 1][1]),
                                         (outPath[id_][0] - outPath[id_ - 1][0]))
                if curHeading != nextHeading:  # add the "id_-1" th element
                    refinedPath.append(outPath[id_ - 1])
                    curHeading = nextHeading  # update the current heading
            refinedPath.append(outPath[-1])

            if self._using_precomputed_maps:
                goal_points = [
                    [
                        float(self.cropped_to_actual[(int(points[0]), int(points[1]))][0]),
                        float(self.cropped_to_actual[(int(points[0]), int(points[1]))][1]),
                    ]
                    for points in refinedPath
                ]
            else:
                goal_points = [
                    [
                        (points[0] + math.ceil(self.bound[0] / self.grid_length)) * self.grid_length,
                        (points[1] + math.ceil(self.bound[2] / self.grid_length)) * self.grid_length,
                    ]
                    for points in refinedPath
                ]

            # load the to goal, but remove/exclude the 1st point, which is the initial position
            self.all_uavs[agentIdx].goal = [
                point for point in goal_points
                if not np.array_equal(np.array(point), self.all_uavs[agentIdx].ini_pos)
            ]  # if not np.array_equal(np.array(points), self.all_agents[agentIdx].ini_pos)

            self.all_uavs[agentIdx].waypoints = deepcopy(self.all_uavs[agentIdx].goal)

            # load the to goal but we include the initial position
            goalPt_withini = goal_points

            self.all_uavs[agentIdx].ref_line = LineString(goalPt_withini)
            # ---------------- end of using random initialized agent position for traffic flow ---------

            self.all_uavs[agentIdx].ref_line_segments = {}
            # Iterate over line coordinates and create line segments
            for i in range(len(self.all_uavs[agentIdx].ref_line.coords) - 1):
                start_point = self.all_uavs[agentIdx].ref_line.coords[i]
                end_point = self.all_uavs[agentIdx].ref_line.coords[i + 1]
                segment = LineString([start_point, end_point])
                self.all_uavs[agentIdx].ref_line_segments[(start_point, end_point)] = segment

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            # this is the initialized heading.
            self.all_uavs[agentIdx].heading = math.atan2(self.all_uavs[agentIdx].goal[0][1] -
                                                         self.all_uavs[agentIdx].pos[1],
                                                         self.all_uavs[agentIdx].goal[0][0] -
                                                         self.all_uavs[agentIdx].pos[0])

            # random_spd = random.randint(1, self.all_agents[agentIdx].maxSpeed)  # initial speed is randomly picked from 1 to max speed
            # random_spd = random.randint(1, 3)  # initial speed is randomly picked from 1 to max speed
            # random_spd = 1  # we fixed a initialized spd
            random_spd = 0  # we fixed a initialized spd
            self.all_uavs[agentIdx].vel = np.array([random_spd * math.cos(self.all_uavs[agentIdx].heading),
                                                    random_spd * math.sin(self.all_uavs[agentIdx].heading)])
            self.all_uavs[agentIdx].pre_vel = np.array([random_spd * math.cos(self.all_uavs[agentIdx].heading),
                                                        random_spd * math.sin(self.all_uavs[agentIdx].heading)])

            # NOTE: UAV's max speed don't change with time, so when we find it normalized bound, we use max speed
            # the below is the maximum normalized velocity range for map range -1 to 1, and maxSPD = 15m/s
            norm_vel_x_range = [
                -self.normalizer.norm_scale([self.all_uavs[agentIdx].maxSpeed, self.all_uavs[agentIdx].maxSpeed])[0],
                self.normalizer.norm_scale([self.all_uavs[agentIdx].maxSpeed, self.all_uavs[agentIdx].maxSpeed])[0]]
            norm_vel_y_range = [
                -self.normalizer.norm_scale([self.all_uavs[agentIdx].maxSpeed, self.all_uavs[agentIdx].maxSpeed])[1],
                self.normalizer.norm_scale([self.all_uavs[agentIdx].maxSpeed, self.all_uavs[agentIdx].maxSpeed])[1]]

            # ----------------end of initialize normalized velocity, but based on normalized map. map pos_x & pos_y are normalized to [-1, 1]---------------

            # self.all_uavs[agentIdx].observableSpace = self.current_observable_space(self.all_uavs[agentIdx])

            cur_circle = Point(self.all_uavs[agentIdx].pos[0],
                               self.all_uavs[agentIdx].pos[1]).buffer(self.all_uavs[agentIdx].protectiveBound,
                                                                      cap_style='round')
            # # ----------------------- end of random initialized ------------------------------

            agentRefer_dict[(self.all_uavs[agentIdx].pos[0],
                             self.all_uavs[agentIdx].pos[1])] = self.all_uavs[agentIdx].agent_name

            agentsCoor_list.append(self.all_uavs[agentIdx].pos)

        overall_state, norm_overall_state, polygons_list, all_agent_st_pos, all_agent_ed_pos, all_agent_intersection_point_list, \
            all_agent_line_collection, all_agent_mini_intersection_list = self.cur_state_norm_state_v3(agentRefer_dict,
                                                                                                       self.flags[
                                                                                                           "full_observable_critic"])
        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_uavs.items():
                self._draw_start_marker(ax, agent.pos)
                # plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                #                                                         transform=Affine2D().rotate_deg(
                #                                                             math.degrees(agent.heading))),
                #          color=(1.0, 0.84, 0.0, 0.55), markersize=8, zorder=3.4)
                # plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                start_label = plt.text(
                    agent.pos[0],
                    agent.pos[1] - 3,
                    str(agentIdx),
                    ha='center',
                    va='center',
                    color='#4dd0e1',
                    fontsize=6,
                    fontweight='bold',
                    zorder=3.3,
                )
                start_label.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground='black')])
                # plt.text(agent.pos[0], agent.pos[1]+3, agentIdx)

                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                # ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                # ax.add_patch(detec_circle_mat)
                self._draw_destination_marker(ax, agent.goal[-1], agentIdx)

                # # link individual drone's starting position with its goal
                # ini = agent.ini_pos
                # for wp in agent.goal:
                #     plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                #     plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                #     ini = wp
                # plt.plot(agent.goal[-1][0], agent.goal[-1][1], marker='*', color='y', markersize=10)
                # plt.text(agent.goal[-1][0], agent.goal[-1][1], agent.agent_name)

            # draw occupied_poly
            occupied_texture_drawn = self._draw_occupied_poly_texture(ax)
            if not occupied_texture_drawn:
                for one_poly in self.world_map_2D_polyList[0][0]:
                    one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y')
                    one_poly_mat.set_facecolor('#fff7b3')
                    one_poly_mat.set_alpha(0.5)
                    one_poly_mat.set_linewidth(0.5)
                    ax.add_patch(one_poly_mat)

            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                zero_poly_mat.set_edgecolor('#f3e98a')
                zero_poly_mat.set_alpha(0.8)
                zero_poly_mat.set_linewidth(0.8)
                # ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                matp_poly.set_facecolor('#c62828')
                matp_poly.set_alpha(0.6)
                matp_poly.set_linewidth(1.0)
                ax.add_patch(matp_poly)

            # show the nearest building obstacles
            # nearest_buildingPoly_mat = shapelypoly_to_matpoly(nearest_buildingPoly, True, 'g', 'k')
            # ax.add_patch(nearest_buildingPoly_mat)

            # for demo purposes
            # for poly in polygons_list:
            #     if poly.geom_type == "Polygon":
            #         matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
            #         ax.add_patch(matp_poly)
            #     else:
            #         x, y = poly.xy
            # ax.plot(x, y, color='green', linewidth=2, solid_capstyle='round', zorder=3)
            # # Plot each start point
            # for point_deg, point_pos in st_points.items():
            #     ax.plot(point_pos.x, point_pos.y, 'o', color='blue')
            #
            # # Plot each end point
            # for point_deg, point_pos in ed_points.items():
            #     ax.plot(point_pos.x, point_pos.y, 'o', color='green')
            #
            # # Plot the lines of the LineString
            # for lines in line_collection:
            #     x, y = lines.xy
            #     ax.plot(x, y, color='blue', linewidth=2, solid_capstyle='round', zorder=2)
            #
            # # point_counter = 0
            # # # Plot each intersection point
            # # for point in intersection_point_list:
            # #     for ea_pt in point.geoms:
            # #         point_counter = point_counter + 1
            # #         ax.plot(ea_pt.x, ea_pt.y, 'o', color='red')
            #
            # # plot minimum intersection point
            # # for pt_dist, pt_pos in mini_intersection_list.items():
            # for pt_pos in mini_intersection_list:
            #     if pt_pos.type == 'MultiPoint':
            #         for ea_pt in pt_pos.geoms:
            #             ax.plot(ea_pt.x, ea_pt.y, 'o', color='yellow')
            #     else:
            #         ax.plot(pt_pos.x, pt_pos.y, 'o', color='red')

            # for ele in self.spawn_area1_polymat:
            #     ax.add_patch(ele)
            # for ele2 in self.spawn_area2_polymat:
            #     ax.add_patch(ele2)
            # for ele3 in self.spawn_area3_polymat:
            #     ax.add_patch(ele3)
            # for ele4 in self.spawn_area4_polymat:
            #     ax.add_patch(ele4)

            ax.set_xlim(self.bound[0], self.bound[1])
            ax.set_ylim(self.bound[2], self.bound[3])

            plt.xlabel(" ")
            plt.ylabel(" ")
            # ax.set_axis_off()
            # plt.axis('equal')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,)
            plt.show()

        return overall_state, norm_overall_state

    def get_current_agent_nei(self, cur_agent, agentRefer_dict, queue):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        # search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound - 1e-6
        search_distance = self.search_distance
        distance_neigh_agent_list = []
        for agent_idx, agent in self.all_uavs.items():
            if agent.agent_name == cur_agent.agent_name:
                continue
            # get neigh distance
            cur_ts_dist = np.linalg.norm(agent.pos - cur_agent.pos)
            if cur_ts_dist < search_distance:
                if queue:
                    distance_neigh_agent_list.append(
                        (cur_ts_dist, agent_idx, np.array([
                            agent.pos[0], agent.pos[1],
                            agent.vel[0], agent.vel[1],
                            agent.protectiveBound
                        ]))
                    )
                    # Sort the list by distance
                    distance_neigh_agent_list.sort(key=lambda x: x[0])

                    # Create a new ordered dictionary with sorted items
                    cur_agent.surroundingNeighbor = OrderedDict(
                        (neigh_agent_data[1], neigh_agent_data[2]) for neigh_agent_data in distance_neigh_agent_list
                    )
                else:
                    cur_agent.surroundingNeighbor[agent_idx] = np.array([agent.pos[0], agent.pos[1],
                                                                         agent.vel[0], agent.vel[1],
                                                                         agent.protectiveBound])
        return cur_agent.surroundingNeighbor

    def _get_observation_neighbors(self, agent):
        max_neighbors = int(getattr(self, "nearest_neighbor_count", 0) or 0)
        if max_neighbors <= 0:
            return list(agent.surroundingNeighbor.items())
        return list(agent.surroundingNeighbor.items())[:max_neighbors]

    def _get_observation_neighbor_capacity(self):
        configured_count = int(getattr(self, "nearest_neighbor_count", 0) or 0)
        if configured_count <= 0:
            return len(self.all_uavs) - 1
        return min(configured_count, len(self.all_uavs) - 1)

    def cur_state_norm_state_v3(self, agentRefer_dict, full_observable_critic_flag):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        combine_overall_state_p1 = []
        overall_state_p2 = []
        combine_overall_state_p2 = []
        overall_state_p2_radar = []
        combine_overall_state_p2_radar = []
        overall_state_p3 = []

        # prepare normalized output states
        norm_overall_state_p1 = []
        combine_norm_overall_state_p1 = []
        norm_overall_state_p2 = []
        combine_norm_overall_state_p2 = []
        norm_overall_state_p2_radar = []
        combine_norm_overall_state_p2_radar = []
        norm_overall_state_p3 = []

        # record surrounding grids for all drones
        all_agent_st_pos = []
        all_agent_ed_pos = []
        all_agent_intersection_point_list = []
        all_agent_line_collection = []
        all_agent_mini_intersection_list = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle center at their position
        for agentIdx, agent in self.all_uavs.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')

            # get agent's observable space around it
            # obs_grid_time = time.time()
            # self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength_fromv2_flow(self.all_agents[agentIdx])
            # self.all_agents[agentIdx].observableSpace = np.zeros((9))
            # print("generate grid time is {} milliseconds".format((time.time()-obs_grid_time)*1000))
            #
            # identify neighbors (use distance)
            # obs_nei_time = time.time()
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict, queue=True)
            # # print("generate nei time is {} milliseconds".format((time.time() - obs_nei_time) * 1000))

            #region start of create radar (with UAV detection) ------------- #
            # drone_ctr = Point(agent.pos)
            # nearest_buildingPoly_idx = self.allbuildingSTR.nearest(drone_ctr)
            # nearest_buildingPoly = self.world_map_2D_polyList[0][0][nearest_buildingPoly_idx]
            # dist_nearest = drone_ctr.distance(nearest_buildingPoly)
            #
            # # Re-calculate the 20 equally spaced points around the circle
            # st_points = {degree: Point(drone_ctr.x + math.cos(math.radians(degree)) * agent.protectiveBound,
            #                              drone_ctr.y + math.sin(math.radians(degree)) * agent.protectiveBound)
            #                for degree in range(0, 360, 20)}
            # # use centre point as start point
            # st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            # all_agent_st_pos.append(st_points)
            #
            # # radar_dist = (agent.detectionRange / 2) - agent.protectiveBound
            # radar_dist = (agent.detectionRange / 2)
            # # Re-define the polygons and build the STRtree again
            # # polygons_list = [
            # #     Polygon([(1, 1), (1, 3), (3, 3), (3, 1)]),
            # #     Polygon([(2, -1), (2, -3), (4, -3), (4, -1)]),
            # #     Polygon([(-3, -1), (-3, -3), (-1, -3), (-1, -1)]),
            # #     Polygon([(-4, 2), (-4, 4), (-2, 4), (-2, 2)])
            # # ]
            # polygons_list_wBound = self.list_of_occupied_grid_wBound
            # polygons_tree_wBound = self.allbuildingSTR_wBound
            #
            # distances = []
            # intersection_point_list = []
            # mini_intersection_list = []
            # ed_points = {}
            # line_collection = []
            # for point_deg, point_pos in st_points.items():
            #     drone_nearest_flag = -1
            #     building_nearest_flag = -1
            #     # Create a line segment from the circle's center to the point on the perimeter
            #     # end_x = point_pos.x + radar_dist * math.cos(math.radians(point_deg))
            #     # end_y = point_pos.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     # Create a line segment from the circle's center
            #     end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
            #     end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     end_point = Point(end_x, end_y)
            #     ed_points[point_deg] = end_point
            #     min_intersection_pt = end_point  # initialize the min_intersection_pt
            #
            #     # Create the LineString from the start point to the end point
            #     line = LineString([point_pos, end_point])
            #     line_collection.append(line)
            #     # Query the STRtree for polygons that intersect with the line segment
            #     intersecting_polygons = polygons_tree_wBound.query(line)
            #
            #     drone_min_dist = line.length
            #     min_distance = line.length
            #
            #     # Build other drone's position circle, and decide the minimum intersection distance from cur host drone to other drone
            #     for other_agents_idx, others in self.all_agents.items():
            #         if other_agents_idx == agentIdx:
            #             continue
            #         other_circle = Point(others.pos).buffer(agent.protectiveBound)
            #         # Check if the LineString intersects with the circle
            #         if line.intersects(other_circle):
            #             drone_nearest_flag = 0
            #             # Find the intersection point(s)
            #             intersection = line.intersection(other_circle)
            #             # The intersection could be a Point or a MultiPoint
            #             # If it's a MultiPoint, we'll calculate the distance to the first intersection
            #             if intersection.geom_type == 'MultiPoint':
            #                 # Calculate distance from the starting point of the LineString to each intersection point
            #                 drone_perimeter_point = min(intersection.geoms, key=lambda point: drone_ctr.distance(point))
            #
            #             elif intersection.geom_type == 'Point':
            #                 # Calculate the distance from the start of the LineString to the intersection point
            #                 drone_perimeter_point = intersection
            #             elif intersection.geom_type in ['LineString', 'MultiLineString']:
            #                 # The intersection is a line (or part of the line lies on the circle's edge)
            #                 # Find the nearest point on this "intersection line" to the start of the original line
            #                 drone_perimeter_point = nearest_points(drone_ctr, intersection)[1]
            #             elif intersection.geom_type == 'GeometryCollection':
            #                 complex_min_dist = math.inf
            #                 for geom in intersection:
            #                     if geom.geom_type == 'Point':
            #                         dist = drone_ctr.distance(geom)
            #                         if dist < complex_min_dist:
            #                             complex_min_dist = dist
            #                             drone_perimeter_point = geom
            #                     elif geom.geom_type == 'LineString':
            #                         nearest_geom_point = nearest_points(drone_ctr, geom)[1]
            #                         dist = drone_ctr.distance(nearest_geom_point)
            #                         if dist < complex_min_dist:
            #                             complex_min_dist = dist
            #                             drone_perimeter_point = nearest_geom_point
            #             else:
            #                 raise ValueError(
            #                     "Intersection is not a point or multipoint, which is unexpected for LineString and Polygon intersection.")
            #             intersection_point_list.append(drone_perimeter_point)
            #             drone_distance = drone_ctr.distance(drone_perimeter_point)
            #             if drone_distance < drone_min_dist:
            #                 drone_min_dist = drone_distance
            #                 drone_nearest_pt = drone_perimeter_point
            #     # ------------ end of radar check surrounding drone's position -------------------------
            #
            #     # # If there are intersecting polygons, find the nearest intersection point
            #     if len(intersecting_polygons) != 0:  # check if a list is empty
            #         building_nearest_flag = 1
            #         # Initialize the minimum distance to be the length of the line segment
            #         for polygon_idx in intersecting_polygons:
            #             # Check if the line intersects with the building polygon's boundary
            #             if polygons_list_wBound[polygon_idx].geom_type == "Polygon":  # intersection with buildings
            #                 # pass
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
            #                     if intersection_point.type == 'MultiPoint':
            #                         nearest_point = min(intersection_point.geoms,
            #                                             key=lambda point: drone_ctr.distance(point))
            #                     else:
            #                         nearest_point = intersection_point
            #                     intersection_point_list.append(nearest_point)
            #                     distance = drone_ctr.distance(nearest_point)
            #                     # min_distance = min(min_distance, distance)
            #                     if distance < min_distance:
            #                         min_distance = distance
            #                         min_intersection_pt = nearest_point
            #             else:  # possible intersection is not a polygon but a LineString, intersection with boundaries
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     intersection = line.intersection(polygons_list_wBound[polygon_idx])
            #                     if intersection.geom_type == 'Point':
            #                         intersection_distance = intersection.distance(drone_ctr)
            #                         if intersection_distance < min_distance:
            #                             min_distance = intersection_distance
            #                             min_intersection_pt = intersection
            #                     # If it's a line of intersection, add each end points of the intersection line
            #                     elif intersection.geom_type == 'LineString':
            #                         for point in intersection.coords:  # loop through both end of the intersection line
            #                             one_end_of_intersection_line = Point(point)
            #                             intersection_distance = one_end_of_intersection_line.distance(drone_ctr)
            #                             if intersection_distance < min_distance:
            #                                 min_distance = intersection_distance
            #                                 min_intersection_pt = one_end_of_intersection_line
            #                     intersection_point_list.append(min_intersection_pt)
            #
            #         # make sure each look there are only one minimum intersection point
            #         distances.append([min_distance, building_nearest_flag])
            #         mini_intersection_list.append(min_intersection_pt)
            #     else:
            #         # If no intersections, the distance is the length of the line segment
            #         distances.append([line.length, building_nearest_flag])
            #     # ------ end of check intersection on polygon or boundaries ------
            #
            #     # Now we compare the minimum distance of intersection for both polygons and drones
            #     # whichever is short, we will load into the last list.
            #     # distances.append([line.length, building_nearest_flag])  # use this for we don't consider obstacles
            #
            #     if drone_min_dist < min_distance:   # one of the other drone is nearer to cur drone
            #         # replace the minimum distance and minimum intersection point
            #         if len(distances) == 0:
            #             distances.append([drone_min_dist, drone_nearest_flag])
            #         else:
            #             distances[-1] = [drone_min_dist, drone_nearest_flag]
            #         if len(mini_intersection_list) == 0:  # if no building polygon surrounding the host drone, mini_intersection_list will not be populated
            #             mini_intersection_list.append(drone_nearest_pt)
            #         else:
            #             mini_intersection_list[-1] = drone_nearest_pt
            #
            # all_agent_ed_pos.append(ed_points)
            # all_agent_intersection_point_list.append(intersection_point_list)  # this is to save all intersection point for each agent
            # all_agent_line_collection.append(line_collection)
            # all_agent_mini_intersection_list.append(mini_intersection_list)
            # self.all_agents[agentIdx].observableSpace = distances
            #endregion  end of create radar --------------- #

            #region start of create radar only used for buildings no boundary, and return building block's x,y, coord
            # drone_ctr = Point(agent.pos)
            # # current pos normalized
            # norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])
            # # Re-calculate the 20 equally spaced points around the circle
            # # use centre point as start point
            # st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            # all_agent_st_pos.append(st_points)
            #
            # radar_dist = (agent.detectionRange / 2)
            #
            # polygons_list_wBound = self.list_of_occupied_grid_wBound
            # polygons_tree_wBound = self.allbuildingSTR_wBound
            #
            # distances = []
            # radar_info = []
            # intersection_point_list = []  # the current radar prob may have multiple intersections points with other geometries
            # mini_intersection_list = []  # only record the intersection point that is nearest to the drone's centre
            # ed_points = {}
            # line_collection = []  # a collection of all 20 radar's prob
            # for point_deg, point_pos in st_points.items():
            #     # Create a line segment from the circle's center
            #     end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
            #     end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     end_point = Point(end_x, end_y)
            #
            #     # current radar prob heading
            #     cur_prob_heading = math.atan2(end_y-agent.pos[1], end_x-agent.pos[0])
            #
            #     ed_points[point_deg] = end_point
            #     min_intersection_pt = end_point
            #
            #     # Create the LineString from the start point to the end point
            #     line = LineString([point_pos, end_point])
            #     line_collection.append(line)
            #     possible_interaction = polygons_tree_wBound.query(line)
            #     # Check if the LineString intersects with the circle
            #     shortest_dist = math.inf  # initialize shortest distance
            #     sensed_shortest_dist = line.length  # initialize actual prob distance
            #     distances.append(line.length)
            #     if len(possible_interaction) != 0:  # check if a list is empty
            #         building_nearest_flag = 1
            #         # Initialize the minimum distance to be the length of the line segment
            #         for polygon_idx in possible_interaction:
            #             # Check if the line intersects with the building polygon's boundary
            #             if polygons_list_wBound[polygon_idx].geom_type == "Polygon":
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     with warnings.catch_warnings():
            #                         warnings.simplefilter('ignore', category=RuntimeWarning)
            #                         intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
            #                     if intersection_point.geom_type == 'MultiPoint':
            #                         nearest_point = min(intersection_point.geoms,
            #                                             key=lambda point: drone_ctr.distance(point))
            #                     else:
            #                         nearest_point = intersection_point
            #                     intersection_point_list.append(nearest_point)
            #                     sensed_shortest_dist = drone_ctr.distance(nearest_point)
            #                     if sensed_shortest_dist < shortest_dist:
            #                         shortest_dist = sensed_shortest_dist
            #                         min_intersection_pt = nearest_point
            #                         end_point = min_intersection_pt
            #                         # intersection_obstacle_centroid = polygons_list_wBound[polygon_idx].centroid
            #                         # norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos([intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                         # norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                         norm_intersection_obstacle = self.normalizer.nmlz_pos([min_intersection_pt.x, min_intersection_pt.y])
            #                         norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle
            #             else:  # possible intersection is not a polygon but a LineString, meaning it is a boundary line
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     with warnings.catch_warnings():
            #                         warnings.simplefilter('ignore', category=RuntimeWarning)
            #                         intersection = line.intersection(polygons_list_wBound[polygon_idx])
            #                     if intersection.geom_type == 'Point':
            #                         sensed_shortest_dist = intersection.distance(drone_ctr)
            #                         if sensed_shortest_dist < shortest_dist:
            #                             shortest_dist = sensed_shortest_dist
            #                             min_intersection_pt = intersection
            #                             end_point = min_intersection_pt
            #                             # if the radar prob intersects with the boundary line, this is a special type of obstacle, we just store the coordinates of the intersection point.
            #                             intersection_obstacle_centroid = min_intersection_pt
            #                             norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos(
            #                                 [intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                             norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                     # If it's a line of intersection, add each end points of the intersection line
            #                     elif intersection.geom_type == 'LineString':
            #                         for point in intersection.coords:  # loop through both end of the intersection line
            #                             one_end_of_intersection_line = Point(point)
            #                             sensed_shortest_dist = one_end_of_intersection_line.distance(drone_ctr)
            #                             if sensed_shortest_dist < shortest_dist:
            #                                 shortest_dist = sensed_shortest_dist
            #                                 min_intersection_pt = one_end_of_intersection_line
            #                                 end_point = min_intersection_pt
            #                                 # if the radar prob intersects with the boundary line, this is a special type of obstacle, we just store the coordinates of the intersection point.
            #                                 intersection_obstacle_centroid = min_intersection_pt
            #                                 norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos(
            #                                     [intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                                 norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                     intersection_point_list.append(min_intersection_pt)
            #
            #         # make sure each look there are only one minimum intersection point
            #         distances[-1] = sensed_shortest_dist
            #         mini_intersection_list.append(min_intersection_pt)
            #     else:
            #         # If no intersections, the distance is the length of the line segment
            #         distances[-1] = line.length
            #         mini_intersection_list.append(min_intersection_pt)
            #         norm_intersection_obstacle_centroid = np.array([-2, -2])
            #         norm_intersection_delta_pos = np.array([-2, -2])
            #     # radar_info.append(norm_intersection_obstacle_centroid[0])
            #     radar_info.append(norm_intersection_delta_pos[0])
            #     # radar_info.append(norm_intersection_obstacle_centroid[1])
            #     radar_info.append(norm_intersection_delta_pos[1])
            #     self.all_agents[agentIdx].probe_line[point_deg] = LineString([point_pos, end_point])
            # all_agent_ed_pos.append(ed_points)
            # all_agent_intersection_point_list.append(intersection_point_list)
            # all_agent_line_collection.append(line_collection)
            # all_agent_mini_intersection_list.append(mini_intersection_list)
            # # self.all_agents[agentIdx].observableSpace = np.array(distances)
            # self.all_agents[agentIdx].observableSpace = np.array(radar_info)

            #endregion end of create radar only used for buildings no boundary, and return building block's x,y, coord

            #region  ---- start of radar creation (only detect surrounding obstacles) ----
            drone_ctr = Point(agent.pos)

            # Re-calculate the 20 equally spaced points around the circle

            # use centre point as start point
            st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            all_agent_st_pos.append(st_points)

            radar_dist = (agent.detectionRange / 2)

            polygons_list_wBound = self.list_of_occupied_grid_wBound
            polygons_tree_wBound = self.all_buildingSTR_wBound

            distances = []
            radar_info = []
            intersection_point_list = []  # the current radar prob may have multiple intersections points with other geometries
            mini_intersection_list = []  # only record the intersection point that is nearest to the drone's centre
            ed_points = {}
            line_collection = []  # a collection of all 20 radar's prob
            for point_deg, point_pos in st_points.items():
                # Create a line segment from the circle's center
                end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
                end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))

                end_point = Point(end_x, end_y)

                # current radar prob heading
                cur_prob_heading = math.atan2(end_y - agent.pos[1], end_x - agent.pos[0])

                ed_points[point_deg] = end_point
                min_intersection_pt = end_point
                drone_perimeter_point = end_point

                # Create the LineString from the start point to the end point
                line = LineString([point_pos, end_point])
                line_collection.append(line)
                possible_interaction = polygons_tree_wBound.query(line)
                # Check if the LineString intersects with the circle
                shortest_dist = math.inf  # initialize shortest distance
                sensed_shortest_dist = line.length  # initialize actual prob distance
                distances.append(line.length)
                if len(possible_interaction) != 0:  # check if a list is empty
                    building_nearest_flag = 1
                    # Initialize the minimum distance to be the length of the line segment
                    for polygon_idx in possible_interaction:
                        # Check if the line intersects with the building polygon's boundary
                        if polygons_list_wBound[polygon_idx].geom_type == "Polygon":
                            if line.intersects(polygons_list_wBound[polygon_idx]):
                                intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
                                if intersection_point.geom_type == 'MultiPoint':
                                    nearest_point = min(intersection_point.geoms,
                                                        key=lambda check_point: drone_ctr.distance(check_point))
                                else:
                                    nearest_point = intersection_point
                                intersection_point_list.append(nearest_point)
                                sensed_shortest_dist = drone_ctr.distance(nearest_point)
                                if sensed_shortest_dist < shortest_dist:
                                    shortest_dist = sensed_shortest_dist
                                    min_intersection_pt = nearest_point
                        else:  # possible intersection is not a polygon but a LineString, meaning it is a boundary line
                            if line.intersects(polygons_list_wBound[polygon_idx]):
                                intersection = line.intersection(polygons_list_wBound[polygon_idx])
                                if intersection.geom_type == 'Point':
                                    sensed_shortest_dist = intersection.distance(drone_ctr)
                                    if sensed_shortest_dist < shortest_dist:
                                        shortest_dist = sensed_shortest_dist
                                        min_intersection_pt = intersection
                                # If it's a line of intersection, add each end points of the intersection line
                                elif intersection.geom_type == 'LineString':
                                    for point in intersection.coords:  # loop through both end of the intersection line
                                        one_end_of_intersection_line = Point(point)
                                        sensed_shortest_dist = one_end_of_intersection_line.distance(drone_ctr)
                                        if sensed_shortest_dist < shortest_dist:
                                            shortest_dist = sensed_shortest_dist
                                            min_intersection_pt = one_end_of_intersection_line
                                intersection_point_list.append(min_intersection_pt)

                    # make sure each look there are only one minimum intersection point
                    distances[-1] = sensed_shortest_dist
                    mini_intersection_list.append(min_intersection_pt)
                else:
                    # If no intersections, the distance is the length of the line segment
                    distances[-1] = line.length
                    mini_intersection_list.append(min_intersection_pt)
                radar_info.append(sensed_shortest_dist)
                radar_info.append(cur_prob_heading)
            all_agent_ed_pos.append(ed_points)
            all_agent_intersection_point_list.append(intersection_point_list)
            all_agent_line_collection.append(line_collection)
            all_agent_mini_intersection_list.append(mini_intersection_list)
            self.all_uavs[agentIdx].observableSpace = np.array(distances)
            # self.all_agents[agentIdx].observableSpace = np.array(radar_info)
            #endregion ---- end of radar creation (only detect surrounding obstacles) ----

            # -------- normalize radar reading by its maximum range -----
            # for ea_dist_idx, ea_dist in enumerate(self.all_agents[agentIdx].observableSpace):
            #     ea_dist = ea_dist / (self.all_agents[agentIdx].detectionRange / 2)
            #     self.all_agents[agentIdx].observableSpace[ea_dist_idx] = ea_dist
            # -------- end of normalize radar reading by its maximum range -----

            rest_compu_time = time.time()

            host_current_point = Point(agent.pos[0], agent.pos[1])
            cross_err_distance, x_error, y_error, nearest_pt = cross_track_error(host_current_point,
                                                                                 agent.ref_line)  # deviation from the reference line, cross track error
            norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            norm_cross_track_deviation_y = y_error * self.normalizer.y_scale

            # no_norm_cross = np.array([x_error, y_error])
            norm_cross = np.array([norm_cross_track_deviation_x, norm_cross_track_deviation_y])

            # ----- discrete the ref line --------------
            if agent.pre_pos is None:
                cur_heading_rad = agent.heading
            else:
                cur_heading_rad = math.atan2(agent.pos[1] - agent.pre_pos[1], agent.pos[0] - agent.pre_pos[0])

            host_detection_circle = host_current_point.buffer(agent.detectionRange / 2)

            point_b = nearest_points(agent.ref_line, host_current_point)[
                0]  # [0] meaning return must be nearer to the 1st input variable
            dist_to_b = agent.ref_line.project(point_b)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                line_within_circle = agent.ref_line.intersection(host_detection_circle)
            if line_within_circle.length == 0:
                # If there is no intersection, we determine whether this drone is on the left or right of the nearest line segment
                # Identify the closest segment to the nearest point on the line
                segments = list(zip(agent.ref_line.coords[:-1], agent.ref_line.coords[1:]))
                closest_segment = min(segments, key=lambda seg: LineString(seg).distance(point_b))
                # Calculate the side using cross product logic
                A = closest_segment[0]
                B = closest_segment[1]
                C = (agent.pos[0], agent.pos[1])
                # Compute the cross product
                cross_product = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
                if cross_product > 0:  # on left of the closest line segment
                    points_spread = [-2 for _ in range(20)]
                elif cross_product < 0:  # on the right of the closest line segment
                    points_spread = [2 for _ in range(20)]
                else:
                    points_spread = [0 for _ in range(20)]
                    print("point is on the line, which has very low chance, in that case we just assign 0.")
                ref_line_obs = points_spread
                norm_ref_line_obs = np.array(points_spread)

            else:
                # Calculate the total distance we can spread out points from Point B
                total_spread_distance = min(agent.detectionRange / 2, line_within_circle.length)
                # Calculate the interval for the points
                interval = total_spread_distance / 10
                # Get 10 points along the LineString from Point B
                points_spread = [line_within_circle.interpolate(dist_to_b + interval * i) for i in range(1, 11)]
                # For demonstration, return the coordinates of the points
                ref_line_obs = [coord for point in points_spread for coord in point.coords[0]]
                # we normalize these ref_line_coordinates
                norm_ref_line_obs = np.array(
                    [norm_coo for point in points_spread for norm_coo in self.normalizer.scale_pos(point.coords[0])])

            # ----- end of discrete the ref line --------------

            # ------ find nearest neighbour ------
            # loop through neighbors from current time step, and search for the nearest neighbour and its neigh_keys
            nearest_neigh_key = None
            shortest_neigh_dist = math.inf
            for neigh_keys in self.all_uavs[agentIdx].surroundingNeighbor:
                # ----- start of make nei invis when neigh reached their goal -----
                # check if this drone reached their goal yet
                nei_cur_circle = Point(self.all_uavs[neigh_keys].pos[0],
                                       self.all_uavs[neigh_keys].pos[1]).buffer(
                    self.all_uavs[neigh_keys].protectiveBound)

                nei_tar_circle = Point(self.all_uavs[neigh_keys].goal[-1]).buffer(1,
                                                                                  cap_style='round')  # set to [-1] so there are no more reference path
                # when there is no intersection between two geometries, "RuntimeWarning" will appear
                # RuntimeWarning is, "invalid value encountered in intersection"
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    nei_goal_intersect = nei_cur_circle.intersection(nei_tar_circle)
                # if not nei_goal_intersect.is_empty:  # current neigh has reached their goal  # this will affect the drone's state space observation do note of this.
                #     continue  # straight away pass this neigh which has already reached.

                # ----- end of make nei invis when neigh reached their goal -----
                # get distance from host to all the surrounding vehicles
                diff_dist_vec = agent.pos - self.all_uavs[neigh_keys].pos  # host pos vector - intruder pos vector
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)
                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    nearest_neigh_key = neigh_keys

            if nearest_neigh_key == None:
                nearest_neigh_pos = [-2, -2]
                norm_nearest_neigh_pos = nearest_neigh_pos
                delta_nei = nearest_neigh_pos
                norm_delta_nei = np.array(nearest_neigh_pos)
                nearest_neigh_vel = nearest_neigh_pos
                norm_nearest_neigh_vel = nearest_neigh_pos
            else:
                nearest_neigh_pos = self.all_uavs[nearest_neigh_key].pos
                norm_nearest_neigh_pos = self.normalizer.nmlz_pos(nearest_neigh_pos)
                delta_nei = nearest_neigh_pos - agent.pos
                norm_delta_nei = norm_nearest_neigh_pos - self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])
                nearest_neigh_vel = self.all_uavs[nearest_neigh_key].vel
                norm_nearest_neigh_vel = self.normalizer.norm_scale(
                    [nearest_neigh_vel[0], nearest_neigh_vel[1]])  # normalization using scale

            # ------- end if find nearest neighbour ------

            # norm_pos = self.normalizer.scale_pos([agent.pos[0], agent.pos[1]])
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            # norm_vel = self.normalizer.norm_scale([agent.vel[0], agent.vel[1]])  # normalization using scale
            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])  # normalization using min_max

            # norm_acc = self.normalizer.norm_scale([agent.acc[0], agent.acc[1]])
            norm_acc = self.normalizer.nmlz_acc([agent.acc[0], agent.acc[1]])  # norm using min_max

            norm_G = self.normalizer.nmlz_pos([agent.goal[-1][0], agent.goal[-1][1]])
            norm_deltaG = norm_G - norm_pos  # drone's position relative to goal, so is like treat goal as the origin.

            norm_seg = self.normalizer.nmlz_pos([agent.goal[0][0], agent.goal[0][1]])
            norm_delta_segG = norm_seg - norm_pos

            # agent_own = np.array([agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])
            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], nearest_neigh_pos[0],
            #                       nearest_neigh_pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], delta_nei[0], delta_nei[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], delta_nei[0], delta_nei[1],
            #                       nearest_neigh_vel[0], nearest_neigh_vel[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1]]+ref_line_obs+
            #                       [agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       agent.goal[0][0]-agent.pos[0], agent.goal[0][1]-agent.pos[1]])

            # agent_own = np.array([agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_nearest_neigh_pos], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_delta_nei], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_delta_nei, norm_nearest_neigh_vel], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_ref_line_obs, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_acc, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_vel, norm_acc, norm_deltaG], axis=0)

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_delta_segG], axis=0)
            # norm_agent_own = np.concatenate([norm_vel, norm_deltaG], axis=0)

            # ---------- based on 1 Dec 2023, add obs for ref line -----------
            # host_current_point = Point(agent.pos[0], agent.pos[1])
            # cross_err_distance, x_error, y_error = self.cross_track_error(host_current_point, agent.ref_line)  # deviation from the reference line, cross track error
            # norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            # norm_cross_track_deviation_y = y_error * self.normalizer.y_scale
            #
            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], x_error, y_error, cross_err_distance])
            #
            # combine_normXY = math.sqrt(norm_cross_track_deviation_x**2 + norm_cross_track_deviation_y**2)
            # norm_cross = np.array([norm_cross_track_deviation_x, norm_cross_track_deviation_y, combine_normXY])
            #
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_cross], axis=0)
            # ---------- end of based on 1 Dec 2023, add obs for ref line -----------

            other_agents = []
            norm_other_agents = []
            p1_other_agents = []
            p1_norm_other_agents = []
            # p2_just_euclidean_delta = []
            p2_just_neighbour = []
            p2_norm_just_neighbour = []
            nearest_neight = []
            norm_nearest_neigh = []
            observation_neighbors = self._get_observation_neighbors(agent)
            # filling term for no surrounding agent detected
            pre_total_possible_conflict = 0  # total possible conflict between the host drone and the current neighbour
            cur_total_possible_conflict = 0  # total possible conflict between the host drone and the current neighbour
            tcpa = -10
            pre_tcpa = -10
            d_tcpa = -10
            pre_d_tcpa = -10
            include_neigh_count = 0
            if len(observation_neighbors) > 0:  # meaning there is surrounding neighbors around the current agent
                for other_agentIdx, other_agent in observation_neighbors:
                    if other_agentIdx != agent_idx:
                        nei_px = self.all_uavs[other_agentIdx].pos[0]
                        nei_py = self.all_uavs[other_agentIdx].pos[1]
                        delta_host_x = self.all_uavs[other_agentIdx].pos[0] - agent.pos[0]
                        delta_host_y = self.all_uavs[other_agentIdx].pos[1] - agent.pos[1]
                        euclidean_dist = np.linalg.norm(self.all_uavs[other_agentIdx].pos - agent.pos)

                        # norm_delta_pos = self.normalizer.scale_pos([delta_host_x, delta_host_y])
                        norm_nei_pos = self.normalizer.nmlz_pos([self.all_uavs[other_agentIdx].pos[0],
                                                                 self.all_uavs[other_agentIdx].pos[1]])
                        norm_delta_pos = norm_pos - norm_nei_pos  # neigh's position relative to host drone. Host drone as origin.

                        norm_euclidean_dist = np.linalg.norm(norm_delta_pos)

                        nei_goal_diff_x = self.all_uavs[other_agentIdx].goal[-1][0] - agent.pos[0]
                        nei_goal_diff_y = self.all_uavs[other_agentIdx].goal[-1][1] - agent.pos[1]

                        nei_heading = self.all_uavs[other_agentIdx].heading
                        nei_acc = self.all_uavs[other_agentIdx].acc
                        nei_norm_acc = self.normalizer.nmlz_acc([nei_acc[0], nei_acc[1]])

                        cur_neigh_vx = self.all_uavs[other_agentIdx].vel[0]
                        cur_neigh_vy = self.all_uavs[other_agentIdx].vel[1]
                        norm_neigh_vel = self.normalizer.nmlz_vel(
                            [cur_neigh_vx, cur_neigh_vy])  # normalization using min_max
                        cur_neigh_ax = self.all_uavs[other_agentIdx].acc[0]
                        cur_neigh_ay = self.all_uavs[other_agentIdx].acc[1]
                        # norm_neigh_acc = self.normalizer.norm_scale([cur_neigh_ax, cur_neigh_ay])
                        norm_neigh_acc = self.normalizer.nmlz_acc([cur_neigh_ax, cur_neigh_ay])

                        # calculate current t_cpa/d_cpa
                        tcpa, d_tcpa, cur_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                            self.all_uavs[other_agentIdx].pos, agent.pos, self.all_uavs[other_agentIdx].vel,
                            agent.vel, self.all_uavs[other_agentIdx].protectiveBound, agent.protectiveBound,
                            cur_total_possible_conflict)
                        # -------------------------------------------------

                        # calculate previous t_cpa/d_cpa
                        pre_tcpa, pre_d_tcpa, pre_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                            self.all_uavs[other_agentIdx].pre_pos, agent.pre_pos,
                            self.all_uavs[other_agentIdx].pre_vel,
                            agent.pre_vel, self.all_uavs[other_agentIdx].protectiveBound, agent.protectiveBound,
                            pre_total_possible_conflict)
                        # ---------------------------
                        if len(nearest_neight) == 0:
                            # nearest_neight = np.array([delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy, nei_heading])
                            nearest_neight = np.array([delta_host_x, delta_host_y])
                        if len(norm_nearest_neigh) == 0:
                            # norm_nearest_neigh = np.array([norm_delta_pos[0], norm_delta_pos[1], norm_neigh_vel[0], [1]])
                            # norm_nearest_neigh = np.append(norm_nearest_neigh, agent.heading)
                            norm_nearest_neigh = np.array([norm_delta_pos[0], norm_delta_pos[1]])

                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx, cur_neigh_vy])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx, cur_neigh_vy, nei_heading])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx,
                        #                               cur_neigh_vy, nei_acc[0], nei_acc[1], nei_heading])
                        p1_surround_agent = np.array(
                            [delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy, nei_heading])
                        # p1_surround_agent = np.array([nei_px, nei_py, cur_neigh_vx, cur_neigh_vy, nei_goal_diff_x,
                        #                               nei_goal_diff_y, nei_heading])
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.append(p1_norm_surround_agent, agent.heading)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel, nei_norm_acc], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([norm_euclidean_dist]), norm_neigh_vel, nei_norm_acc], axis=0)
                        p1_norm_surround_agent = np.concatenate([norm_delta_pos, norm_neigh_vel], axis=0)
                        p1_norm_surround_agent = np.append(p1_norm_surround_agent, agent.heading)
                        # p1_norm_surround_agent = np.concatenate([norm_nei_pos, norm_neigh_vel, ], axis=0)

                        surround_agent = np.array([[other_agent[0] - agent.pos[0],
                                                    other_agent[1] - agent.pos[1],
                                                    other_agent[-2] - other_agent[0],
                                                    other_agent[-1] - other_agent[1],
                                                    other_agent[2], other_agent[3]]])

                        norm_pos_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1]])

                        norm_G_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[-2] - other_agent[0], other_agent[-1] - other_agent[1]])

                        norm_vel = tuple(self.normalizer.nmlz_vel([other_agent[2], other_agent[3]]))
                        # norm_vel = self.normalizer.nmlz_vel([other_agent[2], other_agent[3]])
                        norm_surround_agent = np.array([list(norm_pos_diff + norm_G_diff + norm_vel)])

                        other_agents.append(surround_agent)
                        norm_other_agents.append(norm_surround_agent)
                        p1_other_agents.append(p1_surround_agent)
                        p1_norm_other_agents.append(p1_norm_surround_agent)
                        # p2_just_euclidean_delta.append(euclidean_dist)
                        p2_just_neighbour.append(p1_surround_agent)
                        p2_norm_just_neighbour.append(p1_norm_surround_agent)
                        include_neigh_count = include_neigh_count + 1
                        # if include_neigh_count > 0:  # only include 2 nearest agents
                        #     break
                overall_state_p3.append(other_agents)
                norm_overall_state_p3.append(norm_other_agents)
            else:
                overall_state_p3.append([np.zeros((1, 6))])
                norm_overall_state_p3.append([np.zeros((1, 6))])

            max_neigh_count = self._get_observation_neighbor_capacity()
            filling_required = max_neigh_count - len(observation_neighbors)
            # filling_value = -2
            filling_value = 0
            # filling_dim = 5
            filling_dim = 5
            for _ in range(filling_required):
                p1_other_agents.append(np.array([filling_value] * filling_dim))
                p1_norm_other_agents.append(np.array([filling_value] * filling_dim))
                p2_just_neighbour.append(np.array([filling_value] * filling_dim))
                p2_norm_just_neighbour.append(np.array([filling_value] * filling_dim))
            all_other_agents = np.concatenate(p1_other_agents)
            norm_all_other_agents = np.concatenate(p1_norm_other_agents)

            all_neigh_agents = np.concatenate(p2_just_neighbour)
            norm_all_neigh_agents = np.concatenate(p2_norm_just_neighbour)

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       tcpa, d_tcpa, pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], agent.heading])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                      agent.acc[0], agent.acc[1], agent.heading])

            self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
                                 agent.goal[-1][0] - agent.pos[0], agent.goal[-1][1] - agent.pos[1], agent.heading])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], agent.heading, delta_nei[0], delta_nei[1]])

            # self_obs = np.array([agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # agent_own = np.concatenate((self_obs, all_other_agents), axis=0)
            agent_own = self_obs
            # agent_own = np.concatenate((self_obs, nearest_neight), axis=0)

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG,
            #                                  (tcpa, d_tcpa, pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            # norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_acc], axis=0)
            # norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1

            norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1
            # norm_self_obs = np.append(norm_self_obs, norm_delta_nei)  # we have to do this because heading dim=1

            # norm_self_obs = np.append(norm_self_obs, norm_nearest_neigh)

            # norm_self_obs = np.concatenate([norm_vel, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_agent_own = np.concatenate((norm_self_obs, norm_all_other_agents), axis=0)
            norm_agent_own = norm_self_obs

            overall_state_p1.append(agent_own)
            # overall_state_p2.append(agent.observableSpace)
            overall_state_p2_radar.append(agent.observableSpace)
            overall_state_p2.append(all_neigh_agents)

            # distances_list = [dist_element[0] for dist_element in agent.observableSpace]
            # mini_index = find_index_of_min_first_element(agent.observableSpace)
            # # distances_list.append(agent.observableSpace[mini_index][1])  # append the one-hot, -1 meaning no detection, 1 is building, 0 is drone
            # overall_state_p2.append(distances_list)

            norm_overall_state_p1.append(norm_agent_own)
            # norm_overall_state_p2.append(agent.observableSpace)
            norm_overall_state_p2_radar.append(agent.observableSpace)
            norm_overall_state_p2.append(norm_all_neigh_agents)

            # norm_overall_state_p2.append(distances_list)

        overall.append(overall_state_p1)
        overall.append(overall_state_p2)
        overall.append(overall_state_p2_radar)
        overall.append(overall_state_p3)
        for list_ in overall_state_p3:
            if len(list_) == 0:
                print("check")
        norm_overall.append(norm_overall_state_p1)
        norm_overall.append(norm_overall_state_p2)
        norm_overall.append(norm_overall_state_p2_radar)
        norm_overall.append(norm_overall_state_p3)
        # print("rest compute time is {} milliseconds".format((time.time() - rest_compu_time) * 1000))
        return overall, norm_overall, polygons_list_wBound, all_agent_st_pos, all_agent_ed_pos, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list


    def display_one_eps_status(self, status_holder, drone_idx, cur_dist_to_goal, cur_step_reward):
        status_holder[drone_idx]['Euclidean_dist_to_goal'] = cur_dist_to_goal
        status_holder[drone_idx]['goal_leading_reward'] = cur_step_reward[0]
        status_holder[drone_idx]['deviation_to_ref_line'] = cur_step_reward[1]
        status_holder[drone_idx]['deviation_to_ref_line_reward'] = cur_step_reward[2]
        status_holder[drone_idx]['near_building_penalty'] = cur_step_reward[3]
        status_holder[drone_idx]['small_step_penalty'] = cur_step_reward[4]
        status_holder[drone_idx]['current_drone_speed'] = cur_step_reward[5]
        status_holder[drone_idx]['addition_near_goal_reward'] = cur_step_reward[6]
        status_holder[drone_idx]['segment_reward'] = cur_step_reward[7]
        status_holder[drone_idx]['neareset_point'] = cur_step_reward[8]
        status_holder[drone_idx]['A'+str(drone_idx)+'_observable space'] = cur_step_reward[9]
        status_holder[drone_idx]['A'+str(drone_idx)+'_heading'] = cur_step_reward[10]
        status_holder[drone_idx]['near_drone_penalty'] = cur_step_reward[11]
        return status_holder

    def ss_reward_Mar(
            self,
            current_ts,
            step_reward_record,
            step_collision_record,
            xy,
            full_observable_critic_flag,
            args,
            evaluation_by_episode,
    ):
        bound_building_check = [False] * 4
        eps_status_holder = [{} for _ in range(len(self.all_uavs))]
        reward, done = [], []
        agent_to_remove = []
        check_goal = [False] * len(self.all_uavs)

        crash_penalty_wall = 20
        big_crash_penalty_wall = 200
        crash_penalty_drone = 1
        reach_target = 20
        survival_penalty = 0
        move_after_reach = -2

        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        dist_to_goal = 0

        for drone_idx, drone_obj in self.all_uavs.items():
            if xy[0] is not None and xy[1] is not None and drone_idx > 0:
                continue
            if xy[0] is not None and xy[1] is not None:
                drone_obj.pos = np.array([xy[0], xy[1]])
                drone_obj.pre_pos = drone_obj.pos

            reached_before_step = drone_obj.reach_target

            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            curPoint = Point(drone_obj.pos)
            if isinstance(drone_obj.removed_goal, np.ndarray):
                host_refline = LineString([drone_obj.removed_goal, drone_obj.goal[0]])
            else:
                host_refline = LineString([drone_obj.ini_pos, drone_obj.goal[0]])

            cross_track_deviation = curPoint.distance(host_refline)

            host_pass_line = LineString([drone_obj.pre_pos, drone_obj.pos])
            host_passed_volume = host_pass_line.buffer(drone_obj.protectiveBound, cap_style='round')
            host_current_circle = Point(drone_obj.pos[0], drone_obj.pos[1]).buffer(drone_obj.protectiveBound)
            host_current_point = Point(drone_obj.pos[0], drone_obj.pos[1])

            nearest_neigh_key = None
            immediate_collision_neigh_key = None
            immediate_tcpa = math.inf
            immediate_d_tcpa = math.inf
            shortest_neigh_dist = math.inf
            cur_total_possible_conflict = 0
            pre_total_possible_conflict = 0
            all_neigh_dist = []
            neigh_relative_bearing = None
            neigh_collision_bearing = None
            for neigh_keys in drone_obj.surroundingNeighbor:
                tcpa, d_tcpa, cur_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                    self.all_uavs[neigh_keys].pos, drone_obj.pos, self.all_uavs[neigh_keys].vel, drone_obj.vel,
                    self.all_uavs[neigh_keys].protectiveBound, drone_obj.protectiveBound, cur_total_possible_conflict)
                pre_tcpa, pre_d_tcpa, pre_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                    self.all_uavs[neigh_keys].pre_pos, drone_obj.pre_pos, self.all_uavs[neigh_keys].pre_vel,
                    drone_obj.pre_vel, self.all_uavs[neigh_keys].protectiveBound, drone_obj.protectiveBound,
                    pre_total_possible_conflict)

                if tcpa >= 0 and tcpa < immediate_tcpa:
                    immediate_tcpa = tcpa
                    immediate_d_tcpa = d_tcpa
                    immediate_collision_neigh_key = neigh_keys
                elif tcpa == -10 and d_tcpa < immediate_tcpa:
                    immediate_tcpa = tcpa
                    immediate_d_tcpa = d_tcpa
                    immediate_collision_neigh_key = neigh_keys

                cur_nei_circle = Point(self.all_uavs[neigh_keys].pos[0],
                                       self.all_uavs[neigh_keys].pos[1]).buffer(
                    self.all_uavs[neigh_keys].protectiveBound)
                cur_nei_tar_circle = Point(self.all_uavs[neigh_keys].goal[-1]).buffer(1, cap_style='round')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    neigh_goal_intersect = cur_nei_circle.intersection(cur_nei_tar_circle)
                if args.mode == 'eval' and evaluation_by_episode is False:
                    if not neigh_goal_intersect.is_empty:
                        continue

                diff_dist_vec = drone_obj.pos - self.all_uavs[neigh_keys].pos
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)

                if self.all_uavs[neigh_keys].reach_target or reached_before_step:
                    euclidean_dist_diff = math.inf
                else:
                    all_neigh_dist.append(euclidean_dist_diff)

                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    neigh_relative_bearing = calculate_bearing(
                        drone_obj.pos[0], drone_obj.pos[1],
                        self.all_uavs[neigh_keys].pos[0], self.all_uavs[neigh_keys].pos[1],
                    )
                    nearest_neigh_key = neigh_keys
                if np.linalg.norm(diff_dist_vec) <= drone_obj.protectiveBound * 2:
                    if args.mode == 'eval' and evaluation_by_episode is False:
                        neigh_collision_bearing = calculate_bearing(
                            drone_obj.pos[0], drone_obj.pos[1],
                            self.all_uavs[neigh_keys].pos[0], self.all_uavs[neigh_keys].pos[1],
                        )
                        if self.all_uavs[neigh_keys].drone_collision \
                                or self.all_uavs[neigh_keys].building_collision \
                                or self.all_uavs[neigh_keys].reach_target \
                                or reached_before_step \
                                or drone_obj.building_collision \
                                or drone_obj.drone_collision \
                                or self.all_uavs[neigh_keys].bound_collision:
                            continue
                        collision_drones.append(neigh_keys)
                        drone_obj.drone_collision = True
                        self.all_uavs[neigh_keys].drone_collision = True
                    else:
                        if self.all_uavs[neigh_keys].reach_target or reached_before_step:
                            pass
                        else:
                            neigh_collision_bearing = calculate_bearing(
                                drone_obj.pos[0], drone_obj.pos[1],
                                self.all_uavs[neigh_keys].pos[0], self.all_uavs[neigh_keys].pos[1],
                            )
                            collision_drones.append(neigh_keys)
                            drone_obj.drone_collision = True

            neigh_count = 0
            flag_previous_nearest_two = 0
            for neigh_keys in drone_obj.pre_surroundingNeighbor:
                for collided_drone_keys in collision_drones:
                    if collided_drone_keys == neigh_keys:
                        flag_previous_nearest_two = 1
                        break
                neigh_count += 1
                if neigh_count > 1:
                    break

            start_of_v1_time = time.time()
            v1_decision = 0
            if not reached_before_step:
                possiblePoly = self.all_buildingSTR.query(host_current_circle)
                for element in possiblePoly:
                    if self.all_buildingSTR.geometries.take(element).intersection(host_current_circle):
                        collide_building = 1
                        v1_decision = collide_building
                        drone_obj.collide_wall_count += 1
                        drone_obj.building_collision = True
                        break
            end_v1_time = (time.time() - start_of_v1_time) * 1000 * 1000

            end_v2_time, end_v3_time, v2_decision, v3_decision = 0, 0, 0, 0
            step_collision_record[drone_idx].append([end_v1_time, end_v2_time, end_v3_time,
                                                     v1_decision, v2_decision, v3_decision])

            tar_circle = Point(drone_obj.goal[-1]).buffer(1, cap_style='round')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                goal_cur_intru_intersect = host_current_circle.intersection(tar_circle)

            wp_reach_threshold_dist = 5
            cur_dist_to_wp = curPoint.distance(Point(drone_obj.waypoints[0]))
            next_wp = np.array(drone_obj.waypoints[0])
            wp_intersect_flag = cur_dist_to_wp < wp_reach_threshold_dist

            rew = 0
            dist_to_goal_coeff = 6
            x_norm, y_norm = self.normalizer.nmlz_pos(drone_obj.pos)
            tx_norm, ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[-1])
            after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])

            dist_left = total_length_to_end_of_line(drone_obj.pos, drone_obj.ref_line)
            dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))

            dist_to_seg_coeff = 0
            seg_reward = dist_to_seg_coeff * 0

            coef_ref_line = 0
            cross_err_distance, x_error, y_error, nearest_pt = cross_track_error(host_current_point, drone_obj.ref_line)
            norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            norm_cross_track_deviation_y = y_error * self.normalizer.y_scale

            if cross_err_distance <= drone_obj.protectiveBound:
                m = (0 - 1) / (drone_obj.protectiveBound - 0)
                dist_to_ref_line = coef_ref_line * (m * cross_err_distance + 1)
            else:
                dist_to_ref_line = -coef_ref_line * 1

            surrounding_collision_penalty = 0

            near_drone_penalty_coef = 10
            near_drone_penalty = 0
            dist_to_penalty_upperbound = 10
            dist_to_penalty_lowerbound = 2.5
            all_neigh_dist.sort()
            c_drone = 1 + (dist_to_penalty_lowerbound / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound))
            m_drone = (0 - 1) / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound)
            if nearest_neigh_key is not None:
                for neigh_dist_idx, shortest_neigh_dist in enumerate(all_neigh_dist):
                    if neigh_dist_idx == 2:
                        break
                    if dist_to_penalty_lowerbound <= shortest_neigh_dist <= dist_to_penalty_upperbound:
                        if neigh_relative_bearing is not None and 90.0 <= neigh_relative_bearing < 270:
                            near_drone_penalty_coef = near_drone_penalty_coef * 2
                        near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * (
                            m_drone * shortest_neigh_dist + c_drone
                        )
                    else:
                        near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * 0
            else:
                near_drone_penalty = near_drone_penalty_coef * 0

            small_step_penalty_coef = 5
            spd_penalty_threshold = drone_obj.maxSpeed / 2
            small_step_penalty_val = (
                spd_penalty_threshold - np.clip(np.linalg.norm(drone_obj.vel), 0, spd_penalty_threshold)
            ) * (1.0 / spd_penalty_threshold)
            small_step_penalty = small_step_penalty_coef * small_step_penalty_val

            near_goal_coefficient = 0
            near_goal_threshold = drone_obj.detectionRange
            actual_after_dist_hg = math.sqrt(
                (drone_obj.pos[0] - drone_obj.goal[-1][0]) ** 2 +
                (drone_obj.pos[1] - drone_obj.goal[-1][1]) ** 2
            )
            near_goal_reward = near_goal_coefficient * (
                (near_goal_threshold - np.clip(actual_after_dist_hg, 0, near_goal_threshold)) / near_goal_threshold
            )

            turningPtConst = drone_obj.detectionRange / 2 - drone_obj.protectiveBound
            dist_array = np.array([dist_info for dist_info in drone_obj.observableSpace])
            ascending_array = np.sort(dist_array)
            min_index = np.argmin(dist_array)
            min_dist = dist_array[min_index]

            near_building_penalty_coef = 3
            turningPtConst = 5
            if turningPtConst == 12.5:
                c = 1.25
            elif turningPtConst == 5:
                c = 2
            m = (0 - 1) / (turningPtConst - drone_obj.protectiveBound)
            if drone_obj.protectiveBound <= min_dist <= turningPtConst:
                near_building_penalty = near_building_penalty_coef * (m * min_dist + c)
            else:
                near_building_penalty = 0

            if reached_before_step:
                check_goal[drone_idx] = True
                agent_to_remove.append(drone_idx)
                rew = rew + reach_target + near_goal_reward
                reward.append(np.array(rew))
                done.append(False)
            elif x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                drone_obj.bound_collision = True
                rew = rew - crash_penalty_wall
                if args.mode == 'eval' and evaluation_by_episode is False:
                    done.append(False)
                else:
                    done.append(True)
                bound_building_check[0] = True
                reward.append(np.array(rew))
            elif collide_building == 1:
                if args.mode == 'eval' and evaluation_by_episode is False:
                    done.append(False)
                else:
                    done.append(True)
                bound_building_check[1] = True
                rew = rew - crash_penalty_wall
                reward.append(np.array(rew))
            elif len(collision_drones) > 0:
                if args.mode == 'eval' and evaluation_by_episode is False:
                    done.append(False)
                else:
                    done.append(True)
                bound_building_check[2] = True
                if neigh_collision_bearing is not None and 90.0 <= neigh_collision_bearing <= 180:
                    crash_penalty_wall = crash_penalty_wall * 2
                rew = rew - crash_penalty_wall
                reward.append(np.array(rew))
                if flag_previous_nearest_two:
                    bound_building_check[3] = True
            elif not goal_cur_intru_intersect.is_empty:
                drone_obj.reach_target = True
                check_goal[drone_idx] = True
                agent_to_remove.append(drone_idx)
                rew = rew + reach_target + near_goal_reward
                reward.append(np.array(rew))
                done.append(False)
            else:
                if xy[0] is None and xy[1] is None:
                    if wp_intersect_flag and len(drone_obj.waypoints) > 1:
                        drone_obj.removed_goal = drone_obj.waypoints.pop(0)
                rew = rew + dist_to_ref_line + dist_to_goal - small_step_penalty + near_goal_reward - near_building_penalty + seg_reward - survival_penalty - near_drone_penalty - surrounding_collision_penalty
                done.append(False)
                reward.append(np.array(rew))

            step_reward_record[drone_idx] = [dist_to_ref_line, rew]
            eps_status_holder = self.display_one_eps_status(
                eps_status_holder,
                drone_idx,
                np.array(after_dist_hg),
                [
                    np.array(dist_to_goal), cross_err_distance, dist_to_ref_line,
                    np.array(near_building_penalty), small_step_penalty,
                    np.linalg.norm(drone_obj.vel), near_goal_reward,
                    seg_reward, nearest_pt, drone_obj.observableSpace,
                    drone_obj.heading, np.array(near_drone_penalty),
                ],
            )

        reach_target_state = [agent.reach_target for _, agent in self.all_uavs.items()]
        goal_state_mismatch = [
            {
                "agent_idx": agent_idx,
                "check_goal": check_goal[agent_idx],
                "reach_target": agent.reach_target,
                "bound_collision": agent.bound_collision,
                "building_collision": agent.building_collision,
                "drone_collision": agent.drone_collision,
                "position": agent.pos.tolist() if isinstance(agent.pos, np.ndarray) else agent.pos,
                "goal": agent.goal[-1].tolist() if isinstance(agent.goal[-1], np.ndarray) else agent.goal[-1],
                "reward": float(reward[agent_idx]) if agent_idx < len(reward) else None,
                "done": bool(done[agent_idx]) if agent_idx < len(done) else None,
                "bound_building_check": list(bound_building_check),
            }
            for agent_idx, agent in self.all_uavs.items()
            if check_goal[agent_idx] != agent.reach_target
        ]
        if goal_state_mismatch:
            raise ValueError(
                "ss_reward_Mar goal-state mismatch detected. check_goal={} reach_target={} details={}".format(
                    check_goal,
                    reach_target_state,
                    goal_state_mismatch,
                )
            )

        if full_observable_critic_flag:
            reward = [np.sum(reward) for _ in reward]

        return reward, done, check_goal, step_reward_record, eps_status_holder, step_collision_record, bound_building_check

    def step(self, actions):
        current_ts = self.step_count
        agentCoorKD_list_update = []
        agentRefer_dict = {}
        coe_a = self.acc_max

        for drone_idx_obj, drone_act in zip(self.all_uavs.items(), actions):
            drone_idx = drone_idx_obj[0]
            drone_obj = drone_idx_obj[1]

            self.all_uavs[drone_idx].pre_surroundingNeighbor = deepcopy(self.all_uavs[drone_idx].surroundingNeighbor)
            self.all_uavs[drone_idx].pre_pos = deepcopy(self.all_uavs[drone_idx].pos)
            self.all_uavs[drone_idx].pre_vel = deepcopy(self.all_uavs[drone_idx].vel)
            self.all_uavs[drone_idx].pre_acc = deepcopy(self.all_uavs[drone_idx].acc)

            if self.mode == 'eval' and self.evaluation_by_episode is False:
                if self.all_uavs[drone_idx].reach_target \
                        or self.all_uavs[drone_idx].bound_collision \
                        or self.all_uavs[drone_idx].building_collision \
                        or self.all_uavs[drone_idx].drone_collision:
                    continue

            ax, ay = drone_act[0], drone_act[1]
            ax = ax * coe_a
            ay = ay * coe_a
            self.all_uavs[drone_idx].acc = np.array([ax, ay])

            curVelx = self.all_uavs[drone_idx].vel[0] + ax * self.time_step
            curVely = self.all_uavs[drone_idx].vel[1] + ay * self.time_step
            next_heading = math.atan2(curVely, curVelx)
            if np.linalg.norm([curVelx, curVely]) >= self.all_uavs[drone_idx].maxSpeed:
                hvx = self.all_uavs[drone_idx].maxSpeed * math.cos(next_heading)
                hvy = self.all_uavs[drone_idx].maxSpeed * math.sin(next_heading)
                self.all_uavs[drone_idx].vel = np.array([hvx, hvy])
            else:
                self.all_uavs[drone_idx].vel = np.array([curVelx, curVely])

            if drone_obj.reach_target:
                delta_x = 0
                delta_y = 0
            else:
                delta_x = self.all_uavs[drone_idx].vel[0] * self.time_step
                delta_y = self.all_uavs[drone_idx].vel[1] * self.time_step

            self.all_uavs[drone_idx].acc = np.array([ax, ay])
            counterCheck_heading = math.atan2(delta_y, delta_x)
            self.all_uavs[drone_idx].heading = counterCheck_heading
            self.all_uavs[drone_idx].pos = np.array([
                self.all_uavs[drone_idx].pos[0] + delta_x,
                self.all_uavs[drone_idx].pos[1] + delta_y,
            ])

            agentCoorKD_list_update.append(self.all_uavs[drone_idx].pos)
            agentRefer_dict[(self.all_uavs[drone_idx].pos[0],
                             self.all_uavs[drone_idx].pos[1])] = self.all_uavs[drone_idx].agent_name

        next_state, next_state_norm, polygons_list, all_agent_st_points, all_agent_ed_points, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list = self.cur_state_norm_state_v3(
            agentRefer_dict,
            self.full_observable_critic,
        )

        step_reward_record = [None] * self.n_agents
        rewards, dones, check_goal, step_reward_record, status_holder, step_collision_record, bound_building_check = self.ss_reward_Mar(
            current_ts,
            step_reward_record,
            self._step_collision_record,
            (None, None),
            self.full_observable_critic,
            self._legacy_args,
            self.evaluation_by_episode,
        )
        self._step_collision_record = step_collision_record  # NOT useful at all, it was used to compare between difference collision check method.
        self.step_count += 1

        info = {
            "raw_next_state_state": next_state,
            "check_goal": check_goal,
            "step_reward_record": step_reward_record,
            "status_holder": status_holder,
            "bound_building_check": bound_building_check,
            "polygons_list": polygons_list,
            "all_agent_st_points": all_agent_st_points,
            "all_agent_ed_points": all_agent_ed_points,
            "all_agent_intersection_point_list": all_agent_intersection_point_list,
            "all_agent_line_collection": all_agent_line_collection,
            "all_agent_mini_intersection_list": all_agent_mini_intersection_list,
        }
        return next_state_norm, next_state, rewards, dones, info
