"""
Authentication Router — Login, Token Generation, and User Verification.
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from backend.core.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory mock user store for quick zero-setup auth (can be mapped to DB user model)
MOCK_USERS_DB = {
    "admin@traffic.gov.in": {
        "username": "admin@traffic.gov.in",
        "hashed_password": hash_password("admin123"),
        "role": "admin",
        "name": "Command Center Admin",
    }
}


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_hours: int = 24


class UserProfile(BaseModel):
    username: str
    role: str
    name: str | None = None


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    """Authenticate operator and return signed JWT."""
    user = MOCK_USERS_DB.get(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(hours=24),
    )
    return TokenResponse(access_token=token, role=user["role"])


@router.get("/me", response_model=UserProfile)
def get_current_user_profile(user: dict = Depends(get_current_user)):
    """Retrieve profile of authenticated user."""
    return UserProfile(username=user.get("sub", "unknown"), role=user.get("role", "operator"), name="Traffic Officer")
