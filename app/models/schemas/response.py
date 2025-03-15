from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')

class StatusMessage(BaseModel):
    """Standard status message schema"""
    status: str
    message: str

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, description="Current page number (1-indexed)")
    page_size: int = Field(50, description="Number of items per page")
    total: int = Field(..., description="Total number of items")
    pages: int = Field(..., description="Total number of pages")

class ResponseModel(GenericModel, Generic[T]):
    """Generic API response model with standardized structure"""
    success: bool = True
    message: str = "Operation successful"
    data: Optional[T] = None
    errors: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def success_response(cls, data: T, message: str = "Operation successful", meta: Optional[Dict[str, Any]] = None) -> "ResponseModel[T]":
        """Create a success response"""
        return cls(
            success=True,
            message=message,
            data=data,
            meta=meta
        )

    @classmethod
    def error_response(cls, message: str = "Operation failed", errors: Optional[List[str]] = None) -> "ResponseModel[T]":
        """Create an error response"""
        return cls(
            success=False,
            message=message,
            errors=errors or []
        )

    @classmethod
    def paginated_response(cls, data: List[T], page: int, page_size: int, total: int, message: str = "Data retrieved successfully") -> "ResponseModel[List[T]]":
        """Create a paginated response"""
        pages = (total + page_size - 1) // page_size  # Ceiling division
        
        return cls(
            success=True,
            message=message,
            data=data,
            meta={
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "pages": pages
                }
            }
        )