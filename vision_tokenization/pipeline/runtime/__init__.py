"""Runtime package for online tokenization execution.

Keep this package init intentionally light. Import concrete submodules such as
``executor`` or ``data`` directly so optional dependencies are only loaded when
their functionality is actually used.
"""

__all__: list[str] = []
