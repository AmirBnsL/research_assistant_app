import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from src.api.database import get_db

security = HTTPBasic()



def get_current_username(credentials: HTTPBasicCredentials = Depends(security), db: Session = Depends(get_db)):
    if credentials.username != "admin" or credentials.password != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
