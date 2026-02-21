"""Central registry for all OSMOSE parameters."""

from __future__ import annotations

import re

from osmose.schema.base import OsmoseField


class ParameterRegistry:
    """Collects all OSMOSE parameter definitions and provides lookup/validation."""

    def __init__(self):
        self._fields: list[OsmoseField] = []
        self._by_pattern: dict[str, OsmoseField] = {}

    def register(self, field: OsmoseField) -> None:
        self._fields.append(field)
        self._by_pattern[field.key_pattern] = field

    def all_fields(self) -> list[OsmoseField]:
        return list(self._fields)

    def fields_by_category(self, category: str) -> list[OsmoseField]:
        return [f for f in self._fields if f.category == category]

    def get_field(self, key_pattern: str) -> OsmoseField | None:
        return self._by_pattern.get(key_pattern)

    def categories(self) -> list[str]:
        seen = []
        for f in self._fields:
            if f.category not in seen:
                seen.append(f.category)
        return seen

    def match_field(self, concrete_key: str) -> OsmoseField | None:
        """Match a concrete key like 'species.k.sp0' to its field pattern."""
        for pattern, field in self._by_pattern.items():
            regex = re.escape(pattern).replace(r"\{idx\}", r"\d+")
            if re.fullmatch(regex, concrete_key):
                return field
        return None

    def validate(self, config: dict[str, object]) -> list[str]:
        """Validate a flat config dict against registered field constraints."""
        errors = []
        for key, value in config.items():
            field = self.match_field(key)
            if field:
                field_errors = field.validate_value(value)
                for e in field_errors:
                    errors.append(f"{key}: {e}")
        return errors
