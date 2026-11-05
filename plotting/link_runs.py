"""Append run folders matching the globs to <target>/.baselines so plotting scripts include them.

Usage (repo root): python plotting/link_runs.py <target_runs_dir> <glob> [<glob> ...]
"""

import os
import sys
from glob import glob

from common_utils import read_baseline_entries


def main():
    target, patterns = sys.argv[1], sys.argv[2:]
    os.makedirs(target, exist_ok=True)
    existing = read_baseline_entries(target)
    new = []
    for pattern in patterns:
        for d in sorted(glob(pattern)):
            if os.path.isdir(d) and d not in existing and d not in new:
                new.append(d)
    with open(os.path.join(target, ".baselines"), "a") as fh:
        fh.writelines(f"{d}\n" for d in new)
    print(f"{target}/.baselines: added {len(new)} run(s)")


if __name__ == "__main__":
    main()
