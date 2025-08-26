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

class RegionCode(str, Enum):
    UnitedStates = 'us'
    UnitedKingdom = 'gb'
    Canada = 'ca'
    Australia = 'au'
    Japan = 'jp'
    SouthKorea = 'kr'
    China = 'cn'
    Taiwan = 'tw'
    HongKong = 'hk'
    Germany = 'de'
    France = 'fr'
    Italy = 'it'
    Spain = 'es'
    India = 'in'
    Brazil = 'br'
    Mexico = 'mx'
    Russia = 'ru'

class LanguageCode(str, Enum):
    English = 'en'
    Chinese = 'zh'
    Japanese = 'ja'
    Korean = 'ko'
    German = 'de'
    French = 'fr'
    Italian = 'it'
    Spanish = 'es'
    Portuguese = 'pt'
    Russian = 'ru'
    Hindi = 'hi'
    Arabic = 'ar'

# Classes
@dataclass
class BaseResponse(Generic[T]):
    status_code: StatusCode
    message: str
    content: Optional[T] = None