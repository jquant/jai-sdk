from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List, Union
from enum import Enum
import sys

if sys.version < '3.8':
    from typing_extensions import Literal
else:
    from typing import Literal


class InsertParams(BaseModel):
    batch_size: int
    filter_name: Optional[str]
    overwrite: bool = False
    max_insert_workers: Optional[int]


class VisionModes(Enum):
    classifier = "classifier"
    dense = "dense"
    conv = "conv"
    avgpool = "avgpool"


class VisionModels(Enum):
    resnet50 = "resnet50"
    resnet18 = "resnet18"
    alexnet = "alexnet"
    squeezenet = "squeezenet"
    vgg16 = "vgg16"
    densenet = "densenet"
    inception = "inception"
    googlenet = "googlenet"
    shufflenet = "shufflenet"
    mobilenet = "mobilenet"
    resnext50_32x4d = "resnext50_32x4d"
    wide_resnet50_2 = "wide_resnet50_2"
    mnasnet = "mnasnet"


# Subclass [BaseImage]
class VisionHyperparams(BaseModel):
    model_name: VisionModels = "resnet50"
    mode: Union[int, VisionModes] = "classifier"
    resize_H: int = Field(224, desciption='height of image resizing', ge=224)
    resize_W: int = Field(224, desciption='width of image resizing', ge=224)


print(VisionHyperparams().dict())
print(VisionHyperparams(model_name="vgg16").dict())
print(VisionHyperparams(model_name="vgg16", mode=-1).dict())
print(VisionHyperparams(model_name="vgg16", mode="dense").json())