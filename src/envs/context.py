from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

class TimeSlot(Enum):
    PRIME_TIME = "prime_time"  # 20:00-22:00
    LATE_NIGHT = "late_night"  # 22:00-00:00
    AFTERNOON = "afternoon"    # 14:00-18:00
    MORNING = "morning"        # 06:00-12:00

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class DayOfWeek(Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

@dataclass
class Context:
    """Programming context for decision making"""
    hour: int # 0-26 airing hour
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season
    # TODO add more context features as needed
    #special_event: Optional[str] = None  # "christmas", "summer_festival", etc.
    #target_audience: str = "general"     # "children", "adults", "seniors"
    #theme_focus: Optional[str] = None    # "women_directors", "local_cinema"


class RightsUrgency(Enum):
    CRITICAL = "critical"  # < 30 days
    HIGH = "high"         # 30-90 days
    MEDIUM = "medium"     # 90-365 days
    LOW = "low"          # > 365 days

class CompetitionContext(Enum):
    PRE_COMPETITIVE = "pre_competitive"    # competitor showing in 14-30 days
    COMPETITIVE_WINDOW = "competitive_window"  # competitor showing in 7-14 days
    POST_COMPETITIVE = "post_competitive"  # competitor showed 7-30 days ago
    CLEAR = "clear"                       # no competitor activity

@dataclass
class Movie:
    """Represents a movie in the catalog"""
    movie_id: str
    title: str
    genre: str
    year: int
    runtime: int
    country: str
    language: str
    director: str
    predicted_audience: float  # From your audience prediction model
    cultural_value: float      # From cultural value calculator
    rights_expiry: datetime
    last_shown: Optional[datetime] = None
    times_shown: int = 0
    acquisition_cost: float = 0.0

@dataclass
class Context:
    """Programming context for decision making"""
    hour: int # 0-26 airing hour
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    season: Season
    # TODO add more context features as needed
    #special_event: Optional[str] = None  # "christmas", "summer_festival", etc.
    #target_audience: str = "general"     # "children", "adults", "seniors"
    #theme_focus: Optional[str] = None    # "women_directors", "local_cinema"

@dataclass
class CuratorFeedback:
    """Feedback from curator on recommendations"""
    movie_id: str
    context: Context
    accepted: bool
   ## TODO add more feedback metrics
   #curator_score: float  # 0-1, curator's rating of the recommendation
   #feedback_notes: Optional[str] = None