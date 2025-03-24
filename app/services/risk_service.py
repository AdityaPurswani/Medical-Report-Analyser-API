import logging
from datetime import datetime
from typing import Dict, Any, List

from app.models.schemas.risk import RiskAnalysisResult, SentenceAnalysis
from app.core.exceptions import ProcessingException

# Import from your existing code
from MedicalRiskExtractor import InitialScorer, set_seed, EnhancedDementiaRiskAnalyzer

logger = logging.getLogger(__name__)

class EnhancedRiskService:
    """Service for medical report risk analysis with sentence-level highlighting"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern to avoid multiple instances"""
        if cls._instance is None:
            cls._instance = super(EnhancedRiskService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the risk analyzer services"""
        if not EnhancedRiskService._initialized:
            logger.info("Initializing EnhancedRiskService")
            try:
                set_seed(42)  # For reproducibility
                
                # Initialize the InitialScorer for faster analysis
                self.scorer = InitialScorer()
                
                # Also initialize the EnhancedDementiaRiskAnalyzer for highlighting
                self.analyzer = EnhancedDementiaRiskAnalyzer()
                
                EnhancedRiskService._initialized = True
                logger.info("Enhanced risk service initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing risk service: {str(e)}")
        
    async def analyze_report(self, report_text: str) -> RiskAnalysisResult:
        """
        Analyze a single medical report for risk assessment with sentence-level highlighting
        
        Args:
            report_text: Text content of the medical report
            
        Returns:
            RiskAnalysisResult: Structured risk assessment results with sentence scores
        """
        self._check_initialization()
        
        try:
            # Clean and prepare the report text
            cleaned_text = self._preprocess_text(report_text)
            
            # Get the base score from InitialScorer
            base_score = self.scorer.analyze_text(cleaned_text)
            
            # Convert to percentage (0-100 scale) and add 12
            adjusted_score = (base_score * 100) + 12
            
            # Ensure score doesn't exceed 100
            risk_score = min(adjusted_score, 100)
            
            # Determine risk level based on adjusted score
            risk_level = 'Low'
            if risk_score >= 75:
                risk_level = 'High'
            elif risk_score >= 40:
                risk_level = 'Moderate'
            
            # Now get sentence-level analysis for highlighting
            sentence_analysis = self._analyze_sentences(cleaned_text)
            
            # Convert to our API schema
            result = RiskAnalysisResult(
                risk_score=round(risk_score, 2),
                risk_level=risk_level,
                processed_at=datetime.now(),
                sentence_analysis=sentence_analysis
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing report: {str(e)}")
            raise ProcessingException(f"Error analyzing report: {str(e)}")
    
    def _analyze_sentences(self, text: str) -> List[SentenceAnalysis]:
        """Analyze individual sentences for highlighting"""
        try:
            # Use the NLP model from InitialScorer's severity analyzer
            doc = self.scorer.severity_analyzer.nlp(text)
            
            # Analyze each sentence
            sentence_results = []
            for i, sent in enumerate(doc.sents):
                # Get the sentence text
                sentence_text = sent.text.strip()
                
                if not sentence_text:
                    continue
                
                # Get severity score for this sentence
                severity_score = self.scorer.severity_analyzer.analyze_severity(sentence_text)
                
                # Convert to 0-100 scale
                importance_score = severity_score * 100
                
                # Create a sentence analysis object
                sentence_results.append(
                    SentenceAnalysis(
                        sentence=sentence_text,
                        importance_score=round(importance_score, 2),
                        position=i
                    )
                )
            
            return sentence_results
        except Exception as e:
            logger.error(f"Error in sentence analysis: {str(e)}")
            return []  # Return empty list on error
    
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
        if not EnhancedRiskService._initialized:
            raise ProcessingException("Enhanced risk service is not properly initialized")