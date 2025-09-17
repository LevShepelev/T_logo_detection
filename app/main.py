from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn, io
from .infer import TBankDetector

app = FastAPI(title="T-Bank Logo Detector")
detector = TBankDetector()  # грузим один раз

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

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 415: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"image/jpeg","image/png","image/bmp","image/webp"}:
            return ErrorResponse(error="Unsupported media type", detail=file.content_type)
        content = await file.read()
        boxes = detector.detect(content)
        return DetectionResponse(detections=[Detection(bbox=BoundingBox(x_min=x1,y_min=y1,x_max=x2,y_max=y2)) for (x1,y1,x2,y2) in boxes])
    except Exception as e:
        return ErrorResponse(error="inference_error", detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
