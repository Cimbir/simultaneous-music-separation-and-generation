from __future__ import annotations

from typing import Any

import requests

from study_analysis.config import Config

_PAGE_SIZE = 1000


def fetch_participants(config: Config) -> list[dict[str, Any]]:
    return _fetch_all(config, "participants", order="started_at.asc")


def fetch_responses(config: Config) -> list[dict[str, Any]]:
    return _fetch_all(config, "responses", order="participant_id.asc,trial_number.asc")


def _fetch_all(config: Config, table: str, *, order: str) -> list[dict[str, Any]]:
    config.require_credentials()
    endpoint = f"{config.supabase_url}/rest/v1/{table}"
    auth = {
        "apikey": config.supabase_service_key,
        "Authorization": f"Bearer {config.supabase_service_key}",
    }
    params = {"select": "*", "order": order}

    rows: list[dict[str, Any]] = []
    while True:
        headers = {**auth, "Range-Unit": "items",
                   "Range": f"{len(rows)}-{len(rows) + _PAGE_SIZE - 1}"}
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            return rows
