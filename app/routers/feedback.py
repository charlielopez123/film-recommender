from fastapi import APIRouter, HTTPException
from app.schemas.feedback import UpdateParametersRequest, UpdateParametersResponse
from app.services.recommender import update_signals_logic, env  # singleton env instance


router = APIRouter()

@router.post(
    "/update-parameters",
    response_model=UpdateParametersResponse,
    summary="Apply user feedback to update recommender parameters"
)
def update_parameters(req: UpdateParametersRequest):
    return update_signals_logic(req)
