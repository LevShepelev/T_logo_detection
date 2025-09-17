# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from .infer import TBankDetector

app = FastAPI(title="T-Bank Logo Detector")
detector = TBankDetector()  # load once

ALLOWED = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def detect_logo(file: UploadFile = File(...)):
    try:
        if file.content_type not in ALLOWED:
            return JSONResponse(
                status_code=415,
                content=ErrorResponse(
                    error="unsupported_media_type",
                    detail=f"got {file.content_type}, expected one of {sorted(ALLOWED)}",
                ).model_dump(),
            )

        content = await file.read()
        boxes = detector.detect(content)
        return DetectionResponse(
            detections=[
                Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2))
                for (x1, y1, x2, y2) in boxes
            ]
        )

    except Exception as e:
        # Log if you want: print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="inference_error", detail=str(e)).model_dump(),
        )
