"""Training jobs and shared response helpers."""
import asyncio
import math
import random
from datetime import datetime, timezone
from typing import List
from soundsense.services.state import server_state

def get_uptime_seconds() -> int:
    started = datetime.fromisoformat(server_state["started_at"])
    return int((datetime.now(timezone.utc) - started).total_seconds())


def int32_hash(s: str) -> int:
    """Simulates the 32-bit signed integer hash behavior of JS string hashing."""
    h = 0
    for char in s:
        h = ((h << 5) - h) + ord(char)
        h &= 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def prediction_pairs(predictions) -> List[tuple]:
    """Turns the API's prediction list into (label, confidence) pairs for the history table."""
    pairs = []
    for p in predictions or []:
        cat = p.get("category")
        label = cat.get("name") if isinstance(cat, dict) else (cat or p.get("label") or "unknown")
        pairs.append((str(label), float(p.get("confidence", 0))))
    return pairs


async def simulate_training(job: dict, total_steps: int):
    for step in range(1, total_steps + 1):
        await asyncio.sleep(0.6)  # 600ms per epoch simulation
        job["currentEpoch"] = step
        job["progress"] = round((step / total_steps) * 100)

        epoch_ratio = step / total_steps
        train_loss = 2.5 * math.exp(-3.5 * epoch_ratio) + 0.08 + (random.random() - 0.5) * 0.03
        val_loss = 2.5 * math.exp(-3.0 * epoch_ratio) + 0.12 + (random.random() - 0.5) * 0.05
        train_acc = min(99.5, 50 + 48 * (1 - math.exp(-4 * epoch_ratio)) + (random.random() - 0.5) * 1.5)
        val_acc = min(98.0, 48 + 46 * (1 - math.exp(-3.5 * epoch_ratio)) + (random.random() - 0.5) * 2.0)

        job["metrics"]["trainLoss"].append(round(train_loss, 4))
        job["metrics"]["valLoss"].append(round(val_loss, 4))
        job["metrics"]["trainAcc"].append(round(train_acc, 2))
        job["metrics"]["valAcc"].append(round(val_acc, 2))

    job["status"] = "completed"
    job["completedAt"] = datetime.now(timezone.utc).isoformat()
    job["progress"] = 100


def train_ast_job(job: dict, fold: int, epochs: int, batch_size: int, learning_rate: float, augment: bool):
    """Run genuine AST fine-tuning in the background and publish measured metrics."""
    try:
        from soundsense.ml.ast_model import train_ast

        def on_epoch(epoch: int, total: int, metrics: dict):
            job["currentEpoch"] = epoch
            job["progress"] = round(epoch * 100 / total)
            for key, value in metrics.items():
                job["metrics"][key].append(value)

        result = train_ast(
            fold=fold, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, augment=augment, progress=on_epoch,
        )
        job["result"] = result
        job["status"] = "completed"
        job["progress"] = 100
        job["completedAt"] = datetime.now(timezone.utc).isoformat()
    except Exception as err:
        job["status"] = "failed"
        job["error"] = str(err)
        job["completedAt"] = datetime.now(timezone.utc).isoformat()


