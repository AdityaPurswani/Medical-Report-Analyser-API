from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

class MedicalReportBase(BaseModel):
    """Base schema for medical report data"""
    text: str = Field(..., description="The complete text of the medical report")

class SentenceAnalysis(BaseModel):
    """Schema for sentence-level analysis"""
    sentence: str = Field(..., description="The text of the sentence")
    importance_score: float = Field(..., description="Importance score (0-100)")
    position: int = Field(..., description="Position of the sentence in the text")
    
    class Config:
        schema_extra = {
            "example": {
                "sentence": "Multiple hyperintense foci are seen in bilateral periventricular white matter.",
                "importance_score": 78.5,
                "position": 4
            }
        }

class RiskAnalysisResult(BaseModel):
    """Schema for risk analysis results with sentence highlighting"""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the analysis result")
    risk_score: float = Field(..., description="Overall risk score (0-100)")
    risk_level: str = Field(..., description="Risk level classification (Low, Moderate, High)")
    processed_at: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")
    sentence_analysis: Optional[List[SentenceAnalysis]] = Field(None, description="Sentence-level analysis for highlighting")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "f8c3de3d-1fea-4d7c-a8b0-29f63c4c3454",
                "risk_score": 65.4,
                "risk_level": "Moderate",
                "processed_at": "2025-03-13T10:30:00",
                "sentence_analysis": [
                    {
                        "sentence": "Multiple hyperintense foci are seen in bilateral periventricular white matter.",
                        "importance_score": 78.5,
                        "position": 4
                    },
                    {
                        "sentence": "No evidence of acute infarction or intracranial hemorrhage.",
                        "importance_score": 45.2,
                        "position": 5
                    }
                ]
            }
        }

class MedicalReportRequest(MedicalReportBase):
    """Schema for medical report analysis request"""
    report_id: Optional[str] = Field(None, description="Optional identifier for the report")
    include_sentence_analysis: Optional[bool] = Field(True, description="Whether to include sentence-level analysis")

class BatchReportRequest(BaseModel):
    """Schema for batch report analysis request"""
    reports: List[MedicalReportRequest] = Field(..., description="List of reports to analyze")