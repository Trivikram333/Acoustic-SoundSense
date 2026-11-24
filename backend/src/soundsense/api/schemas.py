from typing import List, Optional
from pydantic import BaseModel, Field

class TrainRequest(BaseModel):
    modelType: str = "resnet18"
    epochs: int = 50
    batchSize: int = 32
    learningRate: float = 0.001
    folds: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    augmentation: bool = True
    description: str = ""


class SaveModelRequest(BaseModel):
    name: str = "Untitled Checkpoint"
    description: str = ""
    modelType: str = "resnet18"
    accuracy: Optional[float] = None
    fromJobId: Optional[str] = None


class CloudSyncRequest(BaseModel):
    repo: str = "huggingface.co/soundsense/esc50-resnet18"


