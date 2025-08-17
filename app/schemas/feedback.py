# app/schemas/movie.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class SelectedMovieFeedback(BaseModel):
    movie_id: int
    title: str
    director: str
    actors: str
    total_score: float
    signal_contributions: Dict[str, float]
    context_features: List[float]
    selected_signals: Dict[str, bool]

class UnselectedMovieFeedback(BaseModel):
    movie_id: int
    title: str
    signal_contributions: Dict[str, float]

class UpdateParametersRequest(BaseModel):
    timestamp: datetime
    date: Optional[str]          # "YYYY-MM-DD"
    hour: int
    day_of_week: Optional[str]
    season: Optional[str]
    selected_movie: SelectedMovieFeedback
    unselected_movies: List[UnselectedMovieFeedback]
    time_to_select_s: float
    regenerations: int

class UpdateParametersResponse(BaseModel):
    success: bool
    message: Optional[str] = None

