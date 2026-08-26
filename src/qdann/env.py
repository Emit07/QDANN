"""Read settings from the environment, falling back to the repo's untracked .env."""

import os
import pathlib

DOT_ENV = pathlib.Path(__file__).parents[2] / ".env"


def get(name: str) -> str:
    if value := os.environ.get(name):
        return value
    if DOT_ENV.exists():
        for line in DOT_ENV.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip("\"'")
    raise SystemExit(f"{name} is in neither the environment nor {DOT_ENV}")
