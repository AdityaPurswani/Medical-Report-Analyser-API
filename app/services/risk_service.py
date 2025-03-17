import logging
from datetime import datetime
from typing import Dict, Any

from app.models.schemas.risk import RiskAnalysisResult
from app.core.exceptions import ProcessingException

# Import just the InitialScorer from your existing code
from MedicalRiskExtractor import InitialScorer, set_seed

logger = logging.getLogger(__name__)

class RiskService:
    """Lightweight service for medical report risk analysis using just the InitialScorer"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern to avoid multiple instances"""
        if cls._instance is None:
            cls._instance = super(RiskService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the initial scorer (much lighter than full model)"""
        if not RiskService._initialized:
            logger.info("Initializing LightweightRiskService")
            try:
                set_seed(42)  # For reproducibility
                self.scorer = InitialScorer()
                RiskService._initialized = True
                logger.info("Lightweight risk scorer initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing risk scorer: {str(e)}")
        
    async def analyze_report(self, report_text: str) -> RiskAnalysisResult:
        """
        Analyze a single medical report for risk assessment using lightweight scoring
        
        Args:
            report_text: Text content of the medical report
            
        Returns:
            RiskAnalysisResult: Structured risk assessment results
        """
        self._check_initialization()
        
        try:
            # Clean and prepare the report text
            cleaned_text = self._preprocess_text(report_text)
            
            # Analyze the text using just the initial scorer
            score = self.scorer.analyze_text(cleaned_text)
            
            # Convert to percentage (0-100 scale)
            risk_score = score * 100 + 12
            
            # Determine risk level based on score
            risk_level = 'Low'
            if risk_score >= 75:
                risk_level = 'High'
            elif risk_score >= 45:
                risk_level = 'Moderate'
            
            # Convert to our API schema
            result = RiskAnalysisResult(
                risk_score=round(risk_score, 2),
                risk_level=risk_level,
                processed_at=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing report: {str(e)}")
            raise ProcessingException(f"Error analyzing report: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize report text"""
        if not text or not isinstance(text, str):
            return ""
            
        # Basic text cleaning
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove multiple spaces
        
        return text
        
    def _check_initialization(self):
        """Check if the service is properly initialized"""
        if not RiskService._initialized:
            raise ProcessingException("Risk scoring service is not properly initialized")