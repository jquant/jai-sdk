from enum import Enum

__all__ = ["Mode", "PossibleDtypes"]


class PossibleDtypes(str, Enum):
    clip = "Clip"
    clip_text = "ClipText"
    clip_image = "ClipImage"
    image = "Image"
    fasttext = "FastText"
    selfsupervised = "SelfSupervised"
    supervised = "Supervised"
    text = "Text"
    edit = "TextEdit"
    linear = "Linear"
    recommendation = "Recommendation"
    recommendation_system = "RecommendationSystem"
    vector = "Vector"


class Mode(str, Enum):
    complete = "complete"
    summarized = "summarized"
    simple = "simple"
