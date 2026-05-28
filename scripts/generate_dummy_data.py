#!/usr/bin/env python3
"""
scripts/generate_dummy_data.py
------------------------------
Generate a synthetic events.csv + labels.csv suitable for end-to-end testing
of the SensQML pipeline. Real data is IP-protected, so this script lets anyone
exercise the full code path locally.

Usage:
    python scripts/generate_dummy_data.py --n-users 1000 --out-dir data/
"""

import argparse
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = [
    "login", "view_home", "view_product", "search", "add_to_cart",
    "purchase", "return", "support_ticket", "rating_submit", "logout",
    "view_pricing", "click_email", "view_dashboard", "update_settings",
]

# Some events are "engaged" signals — users with more of these are LESS likely to churn
ENGAGED_TOKENS = {"purchase", "rating_submit", "view_dashboard", "click_email"}


def generate_user_events(user_id: str, churned: bool, base_time: datetime) -> list[dict]:
    """Generate a realistic event stream for one user."""
    if churned:
        n_events = random.randint(5, 25)
        engaged_prob = 0.10
    else:
        n_events = random.randint(20, 80)
        engaged_prob = 0.45

    events = []
    t = base_time
    for _ in range(n_events):
        if random.random() < engaged_prob:
            etype = random.choice(list(ENGAGED_TOKENS))
        else:
            etype = random.choice(EVENT_TYPES)
        # Variable gaps to exercise gap-token injection
        t += timedelta(seconds=random.randint(60, 60 * 60 * 24 * 3))
        events.append({
            "user_id": user_id,
            "timestamp": t.isoformat(),
            "event_type": etype,
            "value": round(random.uniform(0, 100), 2) if etype == "purchase" else None,
        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--churn-rate", type=float, default=0.3)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    all_events = []
    labels = []
    for i in range(args.n_users):
        uid = f"user_{i:06d}"
        churned = random.random() < args.churn_rate
        all_events.extend(generate_user_events(uid, churned, base))
        labels.append({"user_id": uid, "label": 1 if churned else 0})

    pd.DataFrame(all_events).to_csv(out_path / "events.csv", index=False)
    pd.DataFrame(labels).to_csv(out_path / "labels.csv", index=False)

    print(f"✓ Generated {len(all_events):,} events for {args.n_users:,} users")
    print(f"✓ Churn rate: {args.churn_rate:.0%}")
    print(f"✓ Wrote {out_path}/events.csv and {out_path}/labels.csv")


if __name__ == "__main__":
    main()
