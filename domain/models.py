from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, Optional
T = TypeVar('T')

# Enum
class StatusCode(Enum):
    WAIT_FOR_PROCESS = 0
    SUCCESS = 1
    OTHER = 2
    CALL_API_FAIL = 3

# Classes
@dataclass
class BaseResponse(Generic[T]):
    status_code: StatusCode
    message: str
    content: Optional[T] = None