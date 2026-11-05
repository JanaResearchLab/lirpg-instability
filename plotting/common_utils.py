import atexit
import fnmatch
import json
import os
import re
import time
from collections import defaultdict
from glob import glob

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

SCORE_BOUNDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "score_bounds.json"
)
RESERVOIR_SIZE = 10000  # EventAccumulator's default per-tag scalar sample size


class RunDataCache:
    """Cache for storing and retrieving run data with timestamp checking."""

    def __init__(
        self,
        cache_dir,
        reload_outdated_cache=True,
        cache_freshness_seconds=300,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "run_data_cache.json")
        self.cached_data = self._load_cache()
        self._dirty = False
        atexit.register(self.save)
        self.reload_outdated_cache = reload_outdated_cache
        # Entries younger than this skip the mtime staleness check.
        self.cache_freshness_seconds = cache_freshness_seconds

    def _load_cache(self):
        print(f"Loading cache from {self.cache_file}...", flush=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        tmp_path = f"{self.cache_file}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.cached_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.cache_file)
        except IOError as e:
            print(f"Warning: Could not save cache file: {e}", flush=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _get_file_modification_time(self, file_path):
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0

    def _is_cache_valid(self, event_path, cache_key):
        if cache_key not in self.cached_data:
            print(f"Cache key {cache_key} not found in cached data.", flush=True)
            return False

        cached_entry = self.cached_data[cache_key]
        # Exactly RESERVOIR_SIZE points means the entry was read with default subsampling.
        if len(
            cached_entry.get("values", [])
        ) == RESERVOIR_SIZE and not cached_entry.get("complete"):
            return False
        if "cached_time" not in cached_entry:
            print(
                f"Cached entry for {cache_key} does not have 'cached_time'.", flush=True
            )
            return False

        file_mod_time = self._get_file_modification_time(event_path)
        cached_time = cached_entry["cached_time"]

        if (
            self.reload_outdated_cache
            and file_mod_time > cached_time
            and (time.time() - cached_time) > self.cache_freshness_seconds
        ):
            print(f"Cache for {cache_key} is outdated.", flush=True)
            return False
        return True

    def _resolve_event_path(self, event_path):
        if os.path.isdir(event_path):
            event_path = glob(os.path.join(event_path, "events.out.tfevents.*"))[0]
        return event_path

    def get_cache_key(self, event_path, metric):
        """Key on run_folder/event_file so the key does not depend on the root directory."""
        event_path = self._resolve_event_path(event_path)
        parts = os.path.normpath(event_path).split(os.sep)
        relative_key = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
        return f"{relative_key}::{metric}"

    def get_cached_data(self, event_path, metric: str = "charts/episodic_return"):
        event_path = self._resolve_event_path(event_path)
        cache_key = self.get_cache_key(event_path, metric)

        if self._is_cache_valid(event_path, cache_key):
            cached_entry = self.cached_data[cache_key]
            return (
                cached_entry["steps"],
                cached_entry["values"],
                cached_entry["episodes"],
            )
        return None

    def cache_data(
        self,
        event_path,
        steps,
        values,
        episodes,
        metric: str = "charts/episodic_return",
    ):
        event_path = self._resolve_event_path(event_path)
        cache_key = self.get_cache_key(event_path, metric)

        self.cached_data[cache_key] = {
            "steps": steps,
            "values": values,
            "episodes": episodes,
            "cached_time": time.time(),
            "complete": True,
        }
        self._dirty = True

    def save(self):
        if not self._dirty:
            return
        self._save_cache()
        self._dirty = False


def load_baseline_paths(runs_dir):
    """Return absolute run folders listed in <runs_dir>/.baselines, following nested files."""
    resolved_paths = []
    _collect_baseline_paths(
        os.path.join(os.path.abspath(runs_dir), ".baselines"),
        resolved_paths,
        seen=set(),
        visited_files=set(),
    )
    return resolved_paths


def _collect_baseline_paths(baselines_file, resolved_paths, seen, visited_files):
    baselines_file = os.path.normpath(baselines_file)
    if baselines_file in visited_files or not os.path.exists(baselines_file):
        return
    visited_files.add(baselines_file)
    base = os.path.dirname(os.path.dirname(baselines_file))
    with open(baselines_file, "r") as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            resolved = os.path.normpath(os.path.join(base, entry))
            if os.path.basename(resolved) == ".baselines":
                if not os.path.isfile(resolved):
                    print(
                        f"[WARNING] nested .baselines not found, skipping: {entry} "
                        f"(resolved: {resolved})",
                        flush=True,
                    )
                    continue
                _collect_baseline_paths(resolved, resolved_paths, seen, visited_files)
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not os.path.isdir(resolved):
                print(
                    f"[WARNING] .baselines entry not found, skipping: {entry} "
                    f"(resolved: {resolved})",
                    flush=True,
                )
                continue
            resolved_paths.append(resolved)


def read_baseline_entries(runs_dir):
    """Return the stripped lines of <runs_dir>/.baselines as a set."""
    baselines_file = os.path.join(runs_dir, ".baselines")
    if not os.path.exists(baselines_file):
        return set()
    with open(baselines_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def baseline_event_paths(baseline_folders, folder_pattern=None, extra_predicate=None):
    """Return event files in baseline folders that pass folder_pattern and extra_predicate."""
    event_paths = []
    for folder in baseline_folders or []:
        if folder_pattern and not fnmatch.fnmatch(
            os.path.basename(folder), folder_pattern
        ):
            continue
        for event_path in sorted(glob(os.path.join(folder, "events.out.tfevents.*"))):
            if extra_predicate is None or extra_predicate(event_path):
                event_paths.append(event_path)
    return sorted(event_paths)


class MultiRunDataCache:
    """RunDataCache that routes each path to the cache of the runs directory that contains it."""

    def __init__(
        self,
        primary_runs_dir,
        primary_cache_dir=None,
        reload_outdated_cache=True,
        cache_freshness_seconds=300,
    ):
        self._opts = dict(
            reload_outdated_cache=reload_outdated_cache,
            cache_freshness_seconds=cache_freshness_seconds,
        )
        self._caches = {}
        self._primary_key = os.path.normpath(os.path.abspath(primary_runs_dir))
        cache_dir = primary_cache_dir or os.path.join(primary_runs_dir, ".cache")
        self._caches[self._primary_key] = RunDataCache(cache_dir, **self._opts)

    def _runs_dir_for(self, path):
        p = os.path.normpath(os.path.abspath(path))
        if os.path.isdir(p):
            return os.path.dirname(p)
        return os.path.dirname(os.path.dirname(p))

    def _cache_for(self, path):
        runs_dir = self._runs_dir_for(path)
        if runs_dir not in self._caches:
            self._caches[runs_dir] = RunDataCache(
                os.path.join(runs_dir, ".cache"), **self._opts
            )
        return self._caches[runs_dir]

    def get_cache_key(self, event_path, metric):
        return self._cache_for(event_path).get_cache_key(event_path, metric)

    def _is_cache_valid(self, event_path, cache_key):
        return self._cache_for(event_path)._is_cache_valid(event_path, cache_key)

    def get_cached_data(self, event_path, metric: str = "charts/episodic_return"):
        return self._cache_for(event_path).get_cached_data(event_path, metric)

    def cache_data(
        self,
        event_path,
        steps,
        values,
        episodes,
        metric: str = "charts/episodic_return",
    ):
        return self._cache_for(event_path).cache_data(
            event_path, steps, values, episodes, metric=metric
        )

    def save(self):
        for cache in self._caches.values():
            cache.save()

    @property
    def cache_file(self):
        return self._caches[self._primary_key].cache_file


def detect_games(runs_dir, include_baselines=True):
    """Return sorted env ids parsed from run folder names, including .baselines entries."""
    games = set()
    for folder in os.listdir(runs_dir):
        if not os.path.isdir(os.path.join(runs_dir, folder)):
            continue
        match = re.match(r"(.+?)__--", folder)
        if match:
            games.add(match.group(1))
    if include_baselines:
        for folder in load_baseline_paths(runs_dir):
            match = re.match(r"(.+?)__--", os.path.basename(folder))
            if match:
                games.add(match.group(1))
    return sorted(games)


method_name_mapping = {
    "random_policy": "Random Policy",
    "practice": "PPO-LIRPG (practice)",
    "practice_ppg": "PPO-LIRPG (practice w/ DCPG)",
    "practice_dcpg": "PPO-LIRPG (practice w/ DCPG)",
    "policy_target_rin": "PPO-LIRPG ($r^\mathrm{in}$) (target $r^\mathrm{in}$)",
}

method_postfix_mapping = {
    "ppg": "DCPG",
    "dcpg": "DCPG",
    "ppg_ent_0.01_vreg_0": "DCPG",
    "ppg_vr_0.0_pr_1.0_tfreq_32": "DCPG",
    "popart": "PopArt",
    "practice_ppg": "practice w/ DCPG",
    "ppg_ent_0.01": "DCPG - $\\beta_v$ tuned",
    "ppg_v_reg_0_ent_0.05": "PPG improved",
    "dcpg_v_reg_0_ent_0.05": "DCPG improved",
    "ppg_ent_0.05": "PPG; ent_coef=0.05",
    "ppg_no_v_reg": "PPG; v_reg=0",
    "ent_0.05": "ent_coef=0.05",
    "unshared": "non-shared",
    "freeze_0.3": "with freezing",
    "scale_rin_dens": "scaled $r^\mathrm{in}$",
    "r_in_if_r_ex": "$r^\mathrm{in}$@$r^\mathrm{ex}$",
    "valcoef_0.01": "scaled value $\lambda_V=0.01$",
    "vcoef_0.05": "scaled value $\lambda_V=0.05$",
    "lambda_0.9_gamma_0.9": "$\\gamma=0.9$",
    "gamma_0.9": "reduced $\\gamma=0.9$",
    "pcgrad": "PCGrad",
    "gradnorm": "GradNorm",
    "target_rin_tau0.02": "target $r^\mathrm{in}$",
    "relu": "ReLU $r^\mathrm{in}$",
}


def shorten_env_name(env_name: str) -> str:
    """Return a three-letter env label (InP and InD for the inverted pendulums)."""
    base = env_name.replace("NoFrameskip-v4", "").replace("-v4", "")
    base = base.replace("NoFrameskip", "")
    if base.startswith("InvertedDoublePendulum"):
        return "InD"
    if base.startswith("InvertedPendulum"):
        return "InP"
    return base[:3]


def method_postfix_mapping_fn(postfix):
    postfix = method_postfix_mapping.get(postfix, postfix)
    if all(x in postfix for x in ["ppg", "ent", "vreg"]):
        ppg_splits = postfix.split("_")
        vr = ppg_splits[4]
        return f"PPG ($\\beta_\\pi=1.0, \\beta_v={vr}, t_{{freq}}=32$)"
    if all(x in postfix for x in ["preg0", "tfreq"]):
        preg_splits = postfix.split("_")
        tfreq = preg_splits[3]
        return f"PPG ($\\beta_\\pi=0.0, \\beta_v=0.0, t_{{freq}}={tfreq}$)"
    if all(x in postfix for x in ["ppg", "vr", "pr", "tfreq"]):
        ppg_splits = postfix.split("_")
        vr = ppg_splits[2]
        pr = ppg_splits[4]
        tfreq = ppg_splits[6]
        return f"PPG ($\\beta_\\pi={pr}, \\beta_v={vr}, t_{{freq}}={tfreq}$)"
    return postfix


methods = (
    "PPO ($r^\mathrm{ex}$)",
    "PPO-LIRPG ($r^\mathrm{ex} + r^\mathrm{in}$)",
    "PPO-LIRPG ($r^\mathrm{in}$)",
    "PPO-LIRPG ($r^\mathrm{in}$) (DCPG)",
)
method_color_mapping = {method: ("#F61115", "#9A54B3", "#1B1B1B") for method in methods}
method_color_mapping["PPO-LIRPG ($r^\mathrm{in}$) (with freezing)"] = (
    "#09cac9",
    "#D67AD1",
    "#1B1B1B",
)


def get_avg_return_from_event_file(
    path: str,
    tag: str = "charts/episodic_return",
    window: int = 100,
    cache: RunDataCache = None,
    max_step: int = None,
):
    if cache and cache._is_cache_valid(path, cache.get_cache_key(path, tag)):
        cached = cache.get_cached_data(path, tag)
        steps, values = np.array(cached[0]), np.array(cached[1])
    else:
        events = get_event_data_from_event_file(path, tag)
        if events is None:
            return None
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        episodes = list(range(1, len(values) + 1))
        if cache is not None:
            cache.cache_data(
                path, steps.tolist(), values.tolist(), episodes, metric=tag
            )
    if max_step is not None:
        mask = steps <= max_step
        values = values[mask]
    if window and len(values) < window:
        return None
    if not window:
        return np.mean(values)
    return np.mean(values[-window:])


def get_event_data_from_event_file(path, tag="charts/episodic_return"):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    try:
        events = ea.Scalars(tag)
    except Exception:
        print(f"[WARNING] No events found for tag '{tag}' in {path}", flush=True)
        return None
    return events


def get_first_n_returns_from_event_file(
    path: str,
    n: int = 10,
    tag: str = "charts/episodic_return",
    cache: RunDataCache = None,
):
    if cache and cache._is_cache_valid(path, cache.get_cache_key(path, tag)):
        values = cache.get_cached_data(path, tag)[1]
    else:
        events = get_event_data_from_event_file(path, tag)
        if events is None:
            return None
        steps = [e.step for e in events]
        values = [e.value for e in events]
        episodes = list(range(1, len(values) + 1))
        if cache is not None:
            cache.cache_data(path, steps, values, episodes, metric=tag)
    if len(values) < n:
        return None
    return list(values[:n])


def compute_iqm(values):
    """Mean of the middle 50% (interquartile mean), trimming 25% from each end."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0:
        return None
    trim = n // 4
    if trim == 0 or n - 2 * trim <= 0:
        return float(np.mean(arr))
    return float(np.mean(arr[trim : n - trim]))


def get_random_policy_scores(
    runs_dir, use_median, window=100, cache: RunDataCache = None, baseline_folders=None
):
    random_policy_scores = defaultdict(list)

    def _add_random_score(full_path, folder):
        if "__--random_policy--__" not in folder:
            return
        match = re.match(r"([^/]+)__--random_policy--__", folder)
        if not match:
            return
        env = match.group(1)
        score = get_avg_return_from_event_file(full_path, window=window, cache=cache)
        if score is not None:
            random_policy_scores[env].append(score)

    for folder in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, folder)
        if os.path.isdir(full_path):
            _add_random_score(full_path, folder)

    for full_path in baseline_folders or []:
        _add_random_score(full_path, os.path.basename(full_path))

    aggregation_fn = np.median if use_median else np.mean
    random_policy_scores = {
        env: aggregation_fn(scores)
        for env, scores in random_policy_scores.items()
        if scores
    }
    cache.save()

    return random_policy_scores


def get_setting_from_config(r_ex, r_in, r_rnd):
    if r_rnd > 0 and r_in == 0:
        if r_ex == 0:
            return "PPO-RND (r_rnd)"
        return "PPO-RND"
    elif r_in > 0 and r_rnd == 0:
        if r_ex == 0:
            return "PPO-LIRPG ($r^\mathrm{in}$)"
        return "PPO-LIRPG ($r^\mathrm{ex} + r^\mathrm{in}$)"
    elif r_ex > 0 and r_in > 0 and r_rnd > 0:
        return "PPO-LIRPG-RND"
    elif r_ex > 0 and r_in == 0 and r_rnd == 0:
        return "PPO ($r^\mathrm{ex}$)"


def map_method_name(method, config=None):
    if method in method_name_mapping:
        return method_name_mapping[method]

    if not method.startswith("tuning"):
        return method

    if "-" in method:
        config_str, postfix = method.split("-", 1)
    else:
        config_str, postfix = method, ""

    if config is None:
        try:
            _, r_ex, r_in, r_rnd = config_str.split("_")
            config = float(r_ex), float(r_in), float(r_rnd)
        except ValueError:
            print(f"Unexpected config format in method: {method}", flush=True)
            return method

    r_ex, r_in, r_rnd = config
    setting = get_setting_from_config(r_ex, r_in, r_rnd)
    if setting:
        if postfix:
            postfix = method_postfix_mapping_fn(postfix)
            method = f"{setting} ({postfix})"
        else:
            method = setting
        return method

    return method


def method_name_to_setting(method_name):
    if "PPO-LIRPG-RND" in method_name:
        return "PPO-LIRPG-RND"
    elif "PPO-LIRPG (r_in)" in method_name:
        return "PPO-LIRPG (r_in)"
    elif "PPO-LIRPG" in method_name:
        return "PPO-LIRPG (r_ex + r_in)"
    elif "PPO-RND (r_rnd)" in method_name:
        return "PPO-RND (r_rnd)"
    elif "PPO-RND" in method_name:
        return "PPO-RND"
    elif "PPO" in method_name:
        return "PPO"
    elif "random_policy" in method_name:
        return "random_policy"
    else:
        return "Other"


def load_score_bounds(path=SCORE_BOUNDS_PATH):
    """Return {env_id: {"min": floor, "max": ceiling, ...}} from score_bounds.json, or {} if it is missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def normalize_method_scores(
    method_paths,
    random_policy_scores,
    map_names: bool = True,
    cache: RunDataCache = None,
    window: int = 100,
    max_step: int = None,
    initial_n: int = 10,
    bounds: dict = None,
):
    """Min-max normalize final scores per game, from the stored floor and ceiling where present."""
    if bounds is None:
        bounds = load_score_bounds()
    normalized_scores = {}
    methods = set()
    for game, method_path in method_paths.items():
        print(f"Game: {game}", flush=True)
        scores = {}
        max_score = float("-inf")
        min_score = float("inf")
        initial_returns = []
        for method in method_path:
            print(f"Processing method: {method}", flush=True)
            method_id = method
            if map_names:
                method = map_method_name(method_id)
            methods.add(method)
            scores[method] = []
            for path in tqdm(
                method_path[method_id], desc=f"Normalizing scores for {method}"
            ):
                avg_return = get_avg_return_from_event_file(
                    path, cache=cache, window=window, max_step=max_step
                )
                if avg_return is None:
                    print(
                        f"[WARNING] {game}: skipping {path}, fewer than {window} episodes or no event data"
                    )
                    continue
                if avg_return > max_score:
                    max_score, max_path = avg_return, path
                if avg_return < min_score:
                    min_score = avg_return
                scores[method].append(avg_return)
                first_returns = get_first_n_returns_from_event_file(
                    path, n=initial_n, cache=cache
                )
                if first_returns is not None:
                    initial_returns.extend(first_returns)

        stored = bounds.get(game, {})
        if "max" in stored:
            if max_score > stored["max"] + 1e-6:
                print(
                    f"[WARNING] {game}: {max_path} ({max_score:.2f}) beats the stored ceiling ({stored['max']:.2f})"
                )
            max_score = max(max_score, stored["max"])
        if "min" in stored:
            min_score = stored["min"]
        elif random_policy_scores and game in random_policy_scores:
            min_score = random_policy_scores[game]
        elif initial_returns:
            min_score = compute_iqm(initial_returns)

        denom = max_score - min_score
        if denom <= 0:
            print(
                f"[WARNING] {game}: max_score ({max_score:.4f}) <= min_score ({min_score:.4f}), "
                f"normalization degenerate. Setting all scores to 0. "
                f"Consider skipping this game with --skip_games."
            )
        for method in scores:
            if method == "Random Policy":
                continue
            if denom <= 0:
                scores[method] = [0.0 for _ in scores[method]]
            else:
                scores[method] = [(s - min_score) / denom for s in scores[method]]
        normalized_scores[game] = scores
    return normalized_scores, methods


def extract_method_name(path):
    """Return the text between the two pairs of -- in the run folder name."""
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    match = re.search(r"--(.*)--", path)
    if match:
        return match.group(1)
    else:
        return "unknown"


def extract_config_from_method(method):
    """Return (r_ex, r_in, r_rnd) parsed from a tuning method name, or None."""
    if method.startswith("tuning"):
        if "-" in method:
            config_str, _ = method.split("-", 1)
        else:
            config_str = method
        try:
            _, r_ex, r_in, r_rnd = config_str.split("_")
            return float(r_ex), float(r_in), float(r_rnd)
        except ValueError:
            print(f"Failed to extract config from method: {method}", flush=True)
    return None


# Shorthands for --skip_methods and --only_methods.
METHOD_PATTERN_ALIASES = {
    "mix": r"*($r^\mathrm{ex} + r^\mathrm{in}$)*",
    "in": r"*($r^\mathrm{in}$)*",
    "ex": r"*($r^\mathrm{ex}$)*",
}


def method_matches_any(method, patterns):
    patterns = [METHOD_PATTERN_ALIASES.get(p, p) for p in patterns]
    return any(fnmatch.fnmatchcase(method, pattern) for pattern in patterns)


def organize_paths_by_method(
    paths,
    map_names: bool = True,
    skip_methods: list = None,
    only_methods: list = None,
    disambiguate_configs: bool = False,
):
    """Group paths by display name, optionally appending the config when several map to one name."""
    methods = defaultdict(list)
    method_to_configs = defaultdict(set)

    extracted_data = []

    for path in paths:
        method = extract_method_name(path)
        config = extract_config_from_method(method) if map_names else None
        mapped_method = map_method_name(method, config) if map_names else method

        if skip_methods and method_matches_any(mapped_method, skip_methods):
            continue

        if only_methods and not method_matches_any(mapped_method, only_methods):
            continue

        config_key = None
        if config:
            if mapped_method == "PPO-LIRPG-RND":
                config_key = f"{config[0]},{config[1]},{config[2]}"
            elif mapped_method.startswith("PPO-LIRPG"):
                config_key = f"{config[0]},{config[1]}"
            elif mapped_method.startswith("PPO-RND"):
                config_key = f"{config[0]},{config[2]}"

        extracted_data.append(
            {
                "path": path,
                "mapped_method": mapped_method,
                "config_key": config_key,
                "config": config,
            }
        )

        if config_key is not None:
            method_to_configs[mapped_method].add(config_key)

    for entry in extracted_data:
        mapped_method = entry["mapped_method"]
        config_key = entry["config_key"]
        path = entry["path"]

        if (
            disambiguate_configs
            and len(method_to_configs[mapped_method]) > 1
            and config_key is not None
        ):
            display_method = f"{mapped_method} ({config_key})"
        else:
            display_method = mapped_method

        methods[display_method].append(path)

    return dict(methods)


def get_all_postfixes(runs_dir):
    paths = glob(os.path.join(runs_dir, "*--tuning_*_*_*--*"))
    postfixes = {""}
    for path in paths:
        folder = os.path.basename(path)
        postfix = folder.split("--tuning_")[-1].split("-")[1]
        postfixes.add("-" + postfix)
    return sorted(postfixes)
