from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

class ProcessingException(Exception):
    """Exception raised when processing medical reports fails"""
    pass

class AuthenticationException(Exception):
    """Exception raised for authentication errors"""
    pass

class RateLimitException(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

def add_exception_handlers(app: FastAPI):
    """Add custom exception handlers to the FastAPI app"""
    
    @app.exception_handler(ProcessingException)
    async def processing_exception_handler(request: Request, exc: ProcessingException):
        """Handle processing exceptions"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": "Processing error",
                "errors": [str(exc)],
                "data": None
            }
        )
    
    @app.exception_handler(AuthenticationException)
    async def auth_exception_handler(request: Request, exc: AuthenticationException):
        """Handle authentication exceptions"""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "message": "Authentication error",
                "errors": [str(exc)],
                "data": None
            }
        )
    
    @app.exception_handler(RateLimitException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
        """Handle rate limit exceptions"""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(exc.retry_after)},
            content={
                "success": False,
                "message": "Rate limit exceeded",
                "errors": [str(exc.message)],
                "data": None
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        errors = []
        for error in exc.errors():
            error_msg = f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            errors.append(error_msg)
            
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "Validation error",
                "errors": errors,
                "data": None
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors"""
        errors = []
        for error in exc.errors():
            error_msg = f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            errors.append(error_msg)
            
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": "Data validation error",
                "errors": errors,
                "data": None
            }
        )