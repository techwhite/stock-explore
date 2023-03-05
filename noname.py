from enum import Enum

# consts
class Const(Enum):
    SMOOTH_KEY = 0


class DebugKey(Enum):
    SCORE = 0,
    DISCARD_REASON = 1,
    STOCK_TYPE = 2,
    SORT_SCORE = 3


class StockType(Enum):
    DISCARD = 0,
    SHARP = 1,
    BAND_INCREASE = 2,
    BAND_DECREASE = 3,
    CONSOLIDATION_INCREASE = 4,
    LONG_VALUE = 5,
    PHASED_INCREASE = 6,
    NONE = 100


class DiscardReason(Enum):
    NONE = 0,
    TOO_FAR = 1,
    INVERSE_MARKET = 2,
    SAME_ALIGN_POLES = 3,
    TOO_LITTLE_POLES = 4,
    POLES_INTERVAL_TOO_SMALL = 5,
    POLES_CHANGE_TOO_SMALL = 6,
    TOO_SHORT_DAY_DATA = 7,
    SHORT_SELLING = 8


class PoleType(Enum):
    LOW = 0,
    HIGH = 1
