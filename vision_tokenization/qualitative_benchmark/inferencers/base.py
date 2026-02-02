"""
Abstract base class for inference backends.

Provides argument validation (required/optional args with defaults) and a
common interface so that VLLMInferencer and HFInferencer are interchangeable.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Union

logger = logging.getLogger(__name__)


class BaseInferencer(ABC):
    """
    Abstract inference backend.

    Subclasses declare the arguments they accept via two class variables:

    - ``REQUIRED_ARGS``: argument names that **must** be supplied.
    - ``OPTIONAL_ARGS``: mapping of argument name -> default value.
      These are applied when the caller does not provide them.

    ``__init__`` validates that every required arg is present, fills in
    defaults for optional args, and warns about unknown kwargs that will be
    ignored.  Validated args are stored in ``self._args`` (a plain dict)
    and also set as instance attributes for convenience.
    """

    REQUIRED_ARGS: ClassVar[List[str]] = []
    OPTIONAL_ARGS: ClassVar[Dict[str, Any]] = {}

    def __init__(self, **kwargs):
        # 1. Check required args
        missing = [arg for arg in self.REQUIRED_ARGS if arg not in kwargs]
        if missing:
            raise ValueError(
                f"{type(self).__name__} missing required argument(s): {', '.join(missing)}"
            )

        # 2. Apply defaults for optional args
        for arg, default in self.OPTIONAL_ARGS.items():
            kwargs.setdefault(arg, default)

        # 3. Warn about unknown kwargs
        known = set(self.REQUIRED_ARGS) | set(self.OPTIONAL_ARGS)
        unknown = set(kwargs) - known
        if unknown:
            warnings.warn(
                f"{type(self).__name__}: ignoring unknown argument(s): {', '.join(sorted(unknown))}",
                stacklevel=3,
            )

        # 4. Store validated args and set as attributes
        self._args: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in known}
        for k, v in self._args.items():
            setattr(self, k, v)

    @abstractmethod
    def run_inference(
        self,
        prompt: Union[str, List[int]],
        return_text: bool = False,
        sampling_tmp: float = 0.3,
        sampling_topp: float = 0.95,
        sampling_max_tok: int = 500,
        sampling_min_tok: int = 3,
        sampling_stop_token_ids: List[int] = None,
        debug: bool = False,
    ) -> dict:
        """Run inference on *prompt* and return a result dict.

        The dict always contains ``"generated_ids"`` (a sequence of token IDs).
        If *return_text* is ``True`` it also contains ``"generated_text"``.
        """
        ...

    @property
    @abstractmethod
    def txt_tokenizer(self):
        """Return the text tokenizer used by this backend (or ``None``)."""
        ...
