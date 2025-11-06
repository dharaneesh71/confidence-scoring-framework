"""
Security utilities for authentication and authorization
"""
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from core.config import settings
import secrets


security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """
    Verify admin credentials using HTTP Basic Auth
    """
    correct_username = secrets.compare_digest(
        credentials.username, 
        settings.ADMIN_USERNAME
    )
    correct_password = secrets.compare_digest(
        credentials.password, 
        settings.ADMIN_PASSWORD
    )
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return True