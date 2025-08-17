from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import infer, feedback

def create_app() -> FastAPI:
    app = FastAPI(title="Film Recommender API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # mount your routers
    app.include_router(infer.router, prefix="/infer", tags=["infer"])
    app.include_router(feedback.router, prefix="",                     tags=["feedback"])

    return app

app = create_app()
