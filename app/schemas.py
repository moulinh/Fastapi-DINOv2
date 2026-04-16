from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    class_name: str
    confidence: float
    probabilities: dict[str, float]

class ImageResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    