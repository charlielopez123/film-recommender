import pandas as pd, pickle, time
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np

from envs.env import TVProgrammingEnvironment
from config import settings
from app.schemas.infer import InferRequest, Recommendation
from app.schemas.feedback import UpdateParametersRequest, UpdateParametersResponse

# Load once at import time
catalog_df = pd.read_parquet("data/whatson_data_with_training_data.parquet")
historical_df = pd.read_pickle("data/historical_data_df.pkl")
with open(settings.aud_model_dir/"rf_model.pkl","rb") as f:
    audience_model = pickle.load(f)

env = TVProgrammingEnvironment(
    movie_catalog=catalog_df,
    historical_df=historical_df,
    audience_model=audience_model
)

signal_names = ["curator","audience","competition","diversity","novelty","rights"]

def infer_logic(req: InferRequest) -> JSONResponse:
    start = time.time()
    context_f, context, air_date = env.get_context_features_from_date_hour(req.date, req.hour)
    env.get_available_movies(air_date, context)
    print(f"Number of available movies: {len(env.available_movies)}")

    recommended, top5_idx, top5_scores, w_tilde, movies, X_cands = env.recommend_n_films(context, air_date)

    weights_list = w_tilde.tolist()
    signal_weights = dict(zip(signal_names, weights_list))

    slate = []
    for rank, idx in enumerate(top5_idx):
        meta = env.movie_catalog.loc[movies[idx]]
        contribs = X_cands[idx].tolist()
        weighted_signals = (w_tilde * X_cands[idx]).tolist()
        slate.append(Recommendation(
            movie_id=movies[idx],
            title=meta["title"],
            poster_path=meta["poster_path"],
            director=meta["director"],
            actors=meta["actors"],
            start_rights=meta['start_rights'].to_pydatetime().strftime('%Y-%m-%d'),
            end_rights=meta['end_rights'].to_pydatetime().strftime('%Y-%m-%d'),
            total_score=float(top5_scores[rank]),
            signal_contributions=dict(zip(signal_names, contribs)),
            signal_weights=signal_weights,
            weighted_signals=dict(zip(signal_names, weighted_signals)),
            context_features= list(context_f),
        ))
    elapsed = time.time() - start
    print(f"Inference took {elapsed:.3f}s")
    return JSONResponse(content=jsonable_encoder(slate))


def update_signals_logic(req: UpdateParametersRequest) -> UpdateParametersResponse:
    """
    Called by POST /movies/update-signals.
    - Updates env's internal parameters based on the user's choice.
    - Returns the updated parameters + the other movies' unweighted signals.
    """
    try:
        print('Updating Contextual Thompson Sampler parameters')
        context_features = np.array(req.selected_movie.context_features)
        # 1) apply the update into `env`
        chosen_movie_signals_dict   = req.selected_movie.signal_contributions
        chosen_movie_signals = np.array([v for _,v in  chosen_movie_signals_dict.items()])
        selected_signals_dict   = req.selected_movie.selected_signals
        y_signals = [1 if selected_signals_dict.get(name, False) else 0 for name in signal_names]
        if 1 not in y_signals:
            y_signals = None
        else:
            y_signals = np.array(y_signals)
        
        movie_signals = np.array(req.selected_movie.signal_contributions)
        env.cts.update(
            c_t = context_features,
            s_chosen= chosen_movie_signals,
            r = 1,
            y_signals= y_signals,
        )

        # 2) grab the “others” from the same env
        for unselected_movie in req.unselected_movies:
            movie_signals_dict   = unselected_movie.signal_contributions
            movie_signals = np.array([v for _,v in  movie_signals_dict.items()])
            env.cts.update(
            c_t = context_features,
            s_chosen= movie_signals,
            r = 0,
            )

        param_std = np.sqrt(env.cts.expl_scale / env.cts.h_U)      # same shape as U
        print("Mean param std:", param_std.mean())
        print(f'h_U: {env.cts.h_U.mean()}, h_b: {env.cts.h_b.mean()}')

        # 3) return your Pydantic response
        return UpdateParametersResponse(success=True, message=f"Succesful parameter update !\n Mean param std: {param_std.mean()} \n h_U: {env.cts.h_U.mean()}, h_b: {env.cts.h_b.mean()}")
    except Exception as e:
        # catch/log any errors so the UI can know it failed
        print('Error with parameter update')
        print(e)
        return UpdateParametersResponse(success=False, message=str(e))
