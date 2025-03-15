import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import reports, health
from app.core.exceptions import add_exception_handlers
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    application = FastAPI(
        title="Medical Risk Analysis API",
        description="API for analyzing medical reports and extracting risk assessments",
        version="1.0.0",
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    application.include_router(health.router, prefix="/api", tags=["Health"])
    application.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
    
    # Add custom exception handlers
    add_exception_handlers(application)
    
    @application.on_event("startup")
    async def startup_event():
        """Initialize services and resources on startup"""
        logging.info("Starting Medical Risk Analysis API")
        
        # Preload the risk service singleton to speed up first request
        from app.services.risk_service import RiskService
        try:
            RiskService()
            logging.info("Risk service pre-initialized")
        except Exception as e:
            logging.warning(f"Risk service pre-initialization failed: {str(e)}")
    
    @application.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown"""
        logging.info("Shutting down Medical Risk Analysis API")
    
    return application

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8002, reload=True)