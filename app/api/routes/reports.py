from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from app.models.schemas.risk import (
    MedicalReportRequest, 
    RiskAnalysisResult
)
from app.models.schemas.response import ResponseModel
from app.services.risk_service import RiskService
from app.core.exceptions import ProcessingException

router = APIRouter()

# Dependency to get the lightweight risk service
def get_risk_service():
    """Dependency to get the lightweight risk service instance"""
    return RiskService()

@router.post("/analyze", response_model=ResponseModel[RiskAnalysisResult], status_code=200)
async def analyze_report(
    report: MedicalReportRequest,
    risk_service: RiskService = Depends(get_risk_service)
):
    """
    Analyze a single medical report text for risk assessment using a lightweight approach.
    
    - **text**: The complete text of the medical report
    - **report_id**: Optional identifier for the report
    
    Returns a risk analysis result with score and classification.
    """
    try:
        result = await risk_service.analyze_report(report.text)
        return ResponseModel.success_response(
            data=result,
            message="Medical report successfully analyzed"
        )
    except ProcessingException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")
    
