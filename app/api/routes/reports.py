from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from app.models.schemas.risk import (
    MedicalReportRequest, 
    RiskAnalysisResult
)
from app.models.schemas.response import ResponseModel
from app.services.risk_service import EnhancedRiskService
from app.core.exceptions import ProcessingException

router = APIRouter()

# Dependency to get the risk service
def get_risk_service():
    """Dependency to get the enhanced risk service instance"""
    return EnhancedRiskService()

@router.post("/analyze", response_model=ResponseModel[RiskAnalysisResult], status_code=200)
async def analyze_report(
    report: MedicalReportRequest,
    service: EnhancedRiskService = Depends(get_risk_service)
):
    """
    Analyze a medical report with enhanced sentence-level analysis for highlighting.
    
    - **text**: The complete text of the medical report
    - **report_id**: Optional identifier for the report
    - **include_sentence_analysis**: Whether to include sentence-level analysis for highlighting
    
    Returns a risk analysis result with overall score and sentence-level importance scores.
    """
    try:
        result = await service.analyze_report(report.text)
        return ResponseModel.success_response(
            data=result,
            message="Medical report successfully analyzed with highlighting"
        )
    except ProcessingException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

@router.get("/highlight", response_model=ResponseModel[RiskAnalysisResult])
async def highlight_report(
    text: str = Query(..., description="The medical report text to analyze"),
    service: EnhancedRiskService = Depends(get_risk_service)
):
    """
    Alternative GET endpoint to analyze and highlight a medical report text.
    
    - **text**: The medical report text (passed as a query parameter)
    
    Returns a risk analysis result with sentence-level importance scores for highlighting.
    """
    try:
        result = await service.analyze_report(text)
        return ResponseModel.success_response(
            data=result,
            message="Medical report successfully analyzed with highlighting"
        )
    except ProcessingException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")