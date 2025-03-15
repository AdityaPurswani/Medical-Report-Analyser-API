from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

class MedicalReportBase(BaseModel):
    """Base schema for medical report data"""
    text: str = Field(..., description="The complete text of the medical report")

class RiskAnalysisResult(BaseModel):
    """Schema for risk analysis results"""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the analysis result")
    risk_score: float = Field(..., description="Numerical risk score (0-100)")
    risk_level: str = Field(..., description="Risk level classification (Low, Moderate, High)")
    processed_at: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "f8c3de3d-1fea-4d7c-a8b0-29f63c4c3454",
                "risk_score": 65.4,
                "risk_level": "Moderate",
                "processed_at": "2025-03-13T10:30:00"
            }
        }

class MedicalReportRequest(MedicalReportBase):
    """Schema for medical report analysis request"""
    report_id: Optional[str] = Field(None, description="Optional identifier for the report")

class BatchReportRequest(BaseModel):
    """Schema for batch report analysis request"""
    reports: List[MedicalReportRequest] = Field(..., description="List of reports to analyze")

class BatchRiskResult(BaseModel):
    """Schema for batch risk analysis response"""
    results: List[RiskAnalysisResult]
    processed_count: int
    failed_count: int = 0
    failure_reason: Optional[str] = None

class TrainingRequest(BaseModel):
    """Schema for model training request"""
    reports: List[str] = Field(..., description="List of medical report texts for training")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Optional training parameters")

class TrainingResponse(BaseModel):
    """Schema for model training response"""
    status: str
    message: str
    training_samples: int
    completed_at: datetime = Field(default_factory=datetime.now)