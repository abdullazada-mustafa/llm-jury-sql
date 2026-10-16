"""The Task record shared by tasks.py and hard_tasks.py."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    tier: str
    question: str
    gold: str
    rewrites: tuple[str, ...] = ()
    planted: tuple[tuple[str, str], ...] = ()   # (sql, bug_type)
