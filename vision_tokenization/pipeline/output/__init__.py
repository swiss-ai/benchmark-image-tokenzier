"""Output package for direct writing, spill, and rebuild utilities.

Keep this package init intentionally light. Import concrete submodules such as
``backend``, ``spill``, or ``rebuild`` directly so runtime-only imports do not
pull in unrelated output code.
"""

__all__: list[str] = []
