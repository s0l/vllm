# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable failure classification across executor process boundaries."""

from dataclasses import dataclass
from enum import Enum


class WorkerFailureCode(str, Enum):
    GENERIC = "generic"
    ELASTIC_EXECUTION_PLAN_MISMATCH = "elastic_execution_plan_mismatch"


@dataclass(frozen=True)
class WorkerFailure:
    code: WorkerFailureCode
    remote_type: str
    message: str

    @classmethod
    def from_exception(cls, error: BaseException) -> "WorkerFailure":
        code = getattr(error, "worker_failure_code", WorkerFailureCode.GENERIC)
        try:
            code = WorkerFailureCode(code)
        except ValueError:
            code = WorkerFailureCode.GENERIC
        error_type = type(error)
        return cls(
            code=code,
            remote_type=f"{error_type.__module__}.{error_type.__qualname__}",
            message=str(error),
        )


class WorkerRemoteError(RuntimeError):
    """A typed worker failure reconstructed by the executor parent."""

    def __init__(self, failure: WorkerFailure) -> None:
        self.failure = failure
        super().__init__(f"Worker failed with {failure.remote_type}: {failure.message}")
