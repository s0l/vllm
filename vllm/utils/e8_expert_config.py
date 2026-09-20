# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class E8ArchiveConfig:
    path: str
    resident_fraction: float = 0.35
    max_tokens: int = 1024
    demand_library: str | None = None
    demand_library_sha256: str | None = None
    demand_max_tokens: int = 32
    resident_layout: str | None = None
    resident_layout_sha256: str | None = None

    def __post_init__(self):
        if (
            type(self.path) is not str
            or not Path(self.path).is_absolute()
            or type(self.resident_fraction) is not float
            or not 0.0 <= self.resident_fraction < 1.0
            or type(self.max_tokens) is not int
            or not 1 <= self.max_tokens <= 4096
            or type(self.demand_max_tokens) is not int
            or not 1 <= self.demand_max_tokens <= self.max_tokens
        ):
            raise ValueError("invalid E8 archive configuration")
        if (self.demand_library is None) != (self.demand_library_sha256 is None):
            raise ValueError("E8 demand library path and SHA must be paired")
        if self.demand_library is not None:
            path = Path(self.demand_library)
            digest = self.demand_library_sha256
            if (
                not path.is_absolute()
                or not path.is_file()
                or not isinstance(digest, str)
                or len(digest) != 64
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("invalid E8 demand library identity")
        if (self.resident_layout is None) != (self.resident_layout_sha256 is None):
            raise ValueError("E8 resident layout path and SHA must be paired")
        if self.resident_layout is not None:
            path = Path(self.resident_layout)
            digest = self.resident_layout_sha256
            if (
                not path.is_absolute()
                or not path.is_file()
                or not isinstance(digest, str)
                or len(digest) != 64
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("invalid E8 resident layout identity")

    @classmethod
    def from_options(cls, value):
        if value is None:
            return None
        if (
            not isinstance(value, dict)
            or value.keys()
            - {
                "path",
                "resident_fraction",
                "max_tokens",
                "demand_library",
                "demand_library_sha256",
                "demand_max_tokens",
                "resident_layout",
                "resident_layout_sha256",
            }
            or "path" not in value
        ):
            raise ValueError("invalid E8 archive options")
        return cls(
            value["path"],
            float(value.get("resident_fraction", 0.35)),
            value.get("max_tokens", 1024),
            value.get("demand_library"),
            value.get("demand_library_sha256"),
            value.get("demand_max_tokens", 32),
            value.get("resident_layout"),
            value.get("resident_layout_sha256"),
        )
