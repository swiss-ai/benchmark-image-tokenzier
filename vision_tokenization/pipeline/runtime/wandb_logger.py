"""W&B integration helpers for the distributed tokenization pipeline."""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SimpleWandbLogger",
    "load_wandb_resume_state",
]


def load_wandb_resume_state(
    resume_requested: bool,
    ckpt: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return checkpoint-backed W&B resume state for a valid tokenization resume."""
    if not resume_requested or ckpt is None:
        return None
    state = ckpt.get("wandb")
    if not isinstance(state, dict):
        return None

    run_id = state.get("run_id")
    step = state.get("step")
    if not run_id or step is None:
        logger.warning("Checkpoint is missing W&B resume metadata; starting a fresh W&B run")
        return None

    try:
        step = int(step)
    except (TypeError, ValueError):
        logger.warning("Checkpoint has invalid W&B step %r; starting a fresh W&B run", step)
        return None
    if step < 0:
        logger.warning("Checkpoint has negative W&B step %r; starting a fresh W&B run", step)
        return None

    return {
        "run_id": str(run_id),
        "step": step,
    }


class SimpleWandbLogger:
    """Lightweight W&B logger for rank 0."""

    def __init__(
        self,
        project: str = "vision-tokenization",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        log_interval_seconds: float = 10.0,
        stats_sampling_interval: float = 2.0,
        run_id: Optional[str] = None,
        start_step: int = 0,
    ):
        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags or [],
            config=config or {},
            id=run_id,
            resume="allow" if run_id else None,
            settings=wandb.Settings(x_stats_sampling_interval=stats_sampling_interval),
        )
        self.run_id: str = self._run.id
        self._interval = max(1.0, log_interval_seconds)
        self._last_flush = time.time()
        self._step = max(0, int(start_step))

    def state_dict(self) -> Dict[str, Any]:
        """Return checkpointable W&B resume state."""
        return {
            "run_id": self.run_id,
            "step": self._step,
        }

    def should_log_now(self) -> bool:
        """Return True when the next ``log()`` call would flush to W&B.

        The executor uses this to avoid paying detailed timing/synchronization
        overhead on batches whose metrics would be dropped by the logger's
        interval gate anyway.
        """
        return (time.time() - self._last_flush) >= self._interval

    def log(
        self,
        samples: int,
        tokens: int,
        elapsed_seconds: float,
        image_tokens: int = 0,
        text_tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        timing: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Log absolute totals if the flush interval has elapsed."""
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return
        import wandb

        elapsed = max(0.0, float(elapsed_seconds))
        payload = {
            "samples_processed": samples,
            "tokens_generated": tokens,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "errors": errors,
            "samples_skipped": skipped,
            "samples_per_second": samples / elapsed if elapsed > 0 else 0,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "image_tokens_per_second": image_tokens / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
        }
        if timing:
            payload.update({f"timing/{k}": v for k, v in timing.items()})
        if metrics:
            payload.update(metrics)
        wandb.log(payload, step=self._step)
        self._step += 1
        self._last_flush = now

    def finish(self) -> None:
        import wandb

        wandb.finish()
