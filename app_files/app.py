import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Dict
from envs.env import TVProgrammingEnvironment
from config import settings
import pickle
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=["http://localhost:8080"],  # or ["*"] for quick testing
    allow_credentials=True,
    allow_methods=["*"],      # GET, POST, OPTIONS, etc.
    allow_headers=["*"],      # Content-Type, Authorization, etc.
)

# Load once at startup
catalog_df = pd.read_parquet("data/whatson_data_with_training_data.parquet")
historical_data_df = pd.read_pickle("data/historical_data_df.pkl")
with open(settings.aud_model_dir/"rf_model.pkl", "rb") as f:    # load frozen viewership model
    audience_model = pickle.load(f)

# Load environment
env = TVProgrammingEnvironment(movie_catalog=catalog_df,
                         historical_df= historical_data_df,
                         audience_model=audience_model)


class InferRequest(BaseModel):
    date: str   # "YYYY-MM-DD"
    hour: int

# define your output schema, for clarity and docs
class Recommendation(BaseModel):
    movie_id: int
    title: str
    poster_path: str
    director: str
    actors: str
    start_rights: str
    total_score: float
    signal_contributions: dict[str, float]
    signal_weights: dict[str, float]
    weighted_signals: dict[str, float]

signal_names = ["curator", "audience", "competition", "diversity", "novelty", "rights"]

@app.post("/infer", response_model=List[Recommendation])
def infer(req: InferRequest):
    
    start = time.time()

    context_f, context, air_date = env.get_context_features_from_date_hour(req.date, req.hour)
    env.get_available_movies(air_date, context)
    recommended, top5_idx, top5_scores, w_tilde, movies, X_cands  = env.recommend_n_films(context, air_date)

    # convert global weights once
    weights_list = w_tilde.tolist()
    signal_weights = dict(zip(signal_names, weights_list))

    slate: list[dict] = []
    for rank, idx in enumerate(top5_idx):
        total_score = float(top5_scores[rank])
        contribs = X_cands[idx].tolist()  # length-6 list of floats
        meta = env.movie_catalog.loc[movies[idx]]  
        weighted_signals = list(w_tilde * X_cands[idx])

        slate.append({
            "movie_id":            movies[idx],
            "title":               meta["title"],
            "poster_path":         meta["poster_path"],
            "director":            meta["director"],
            "actors":              meta["actors"],
            "total_score":         total_score,
            "start_rights":        meta['start_rights'].to_pydatetime().strftime('%Y-%m-%d'),
            "end_rights":          meta['end_rights'].to_pydatetime().strftime('%Y-%m-%d'),
            "signal_contributions": dict(zip(signal_names, contribs)),
            "signal_weights":      dict(zip(signal_names, signal_weights)),
            "weighted_signals": dict(zip(signal_names, weighted_signals)),
        })

    end = time.time()
    elapsed_s = end - start
    print(f"Elapsed: {elapsed_s:.3f} s")

    # JSON‐encode and return
    return JSONResponse(content=jsonable_encoder(slate))
