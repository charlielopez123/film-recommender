# app/routers/infer.py
from fastapi import APIRouter
from app.schemas.infer import InferRequest, Recommendation
from app.services.recommender import infer_logic
from typing import List

router = APIRouter()

@router.post("/", response_model=List[Recommendation])
def do_infer(req: InferRequest):
    return infer_logic(req)