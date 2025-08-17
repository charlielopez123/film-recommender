from pydantic import BaseModel
from typing import List, Dict, Optional

class InferRequest(BaseModel):
    date: str
    hour: int

class Recommendation(BaseModel):
    movie_id: int
    title: str
    poster_path: Optional[str]
    director: Optional[str]
    actors: Optional[str]
    start_rights: Optional[str]
    end_rights: Optional[str]
    total_score: float
    signal_contributions: Dict[str, float]
    signal_weights: Dict[str, float]
    weighted_signals: Dict[str, float]
    context_features: List[float]