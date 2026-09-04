"""Cameras router — register and list cameras."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.camera import Camera
from backend.schemas.camera import CameraCreate, CameraOut

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("/", response_model=list[CameraOut])
def list_cameras(deployment: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Camera).filter(Camera.is_active == True)
    if deployment:
        query = query.filter(Camera.deployment == deployment)
    return query.all()


@router.post("/", response_model=CameraOut, status_code=201)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    existing = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Camera {payload.camera_id} already exists.")
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return camera
