from fastapi import APIRouter, Depends
from app.models.schemas.response import ResponseModel, StatusMessage
from app.services.risk_service import RiskService
from app.config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=ResponseModel[StatusMessage])
async def health_check():
    """
    Basic health check for the API
    """
    return ResponseModel.success_response(
        data=StatusMessage(
            status="ok",
            message="Service is running"
        ),
        message="API is operational"
    )

@router.get("/health/service", response_model=ResponseModel[StatusMessage])
async def service_health(
    risk_service: RiskService = Depends(lambda: RiskService())
):
    """
    Deep health check that verifies the risk service is initialized
    """
    try:
        # Simple check to verify service initialization
        risk_service._check_initialization()
        
        return ResponseModel.success_response(
            data=StatusMessage(
                status="ok",
                message="Risk service is ready"
            ),
            message="All services are operational"
        )
    except Exception as e:
        return ResponseModel.error_response(
            message="Risk service is not ready",
            errors=[str(e)]
        )