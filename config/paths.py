import os
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESOURCE_ENV_VAR = "MARL_RESOURCE_FILE"


def _expand_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(raw_path)))
    return Path(expanded)


def resolve_path(raw_path: str, base_dir: Optional[Path] = None) -> str:
    path = _expand_path(raw_path)
    if not path.is_absolute():
        root = base_dir or PROJECT_ROOT
        path = root / path
    return str(path.resolve())


def resolve_resource_path(config: dict, config_path: Optional[str] = None) -> dict:
    paths_cfg = dict(config.get("paths", {}))
    env_cfg = dict(config.get("env", {}))

    config_dir = Path(config_path).resolve().parent if config_path else PROJECT_ROOT
    env_var_name = paths_cfg.get("resource_env_var", DEFAULT_RESOURCE_ENV_VAR)
    resource_value = os.getenv(env_var_name) or paths_cfg.get("resource_file")

    paths_cfg["project_root"] = str(PROJECT_ROOT)
    paths_cfg["config_dir"] = str(config_dir)
    paths_cfg["resource_env_var"] = env_var_name

    if resource_value:
        resolved_resource = resolve_path(resource_value, base_dir=config_dir)
        paths_cfg["resource_file"] = resolved_resource
        env_cfg["resource_file"] = resolved_resource

    config["paths"] = paths_cfg
    config["env"] = env_cfg
    return config


def create_training_run_dirs(config: dict) -> dict:
    paths_cfg = dict(config.get("paths", {}))
    exp_name = config.get("exp_name", "default_exp")

    checkpoint_root = resolve_path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    timestamp = datetime.now().strftime("%d%m%y_%H_%M_%S")

    run_dir = Path(checkpoint_root) / exp_name / timestamp
    plot_dir = run_dir / "toplot"

    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg["checkpoint_dir"] = str(run_dir)
    paths_cfg["plot_dir"] = str(plot_dir)

    config["paths"] = paths_cfg
    return config
