from enum import Enum

__all__ = ["Mode", "PossibleDtypes"]


class PossibleDtypes(str, Enum):
    image = "Image"
    fasttext = "FastText"
    selfsupervised = "SelfSupervised"
    supervised = "Supervised"
    text = "Text"
    edit = "TextEdit"
    recommendation = "Recommendation"
    recommendation_system = "RecommendationSystem"
    vector = "Vector"


class Mode(str, Enum):
    complete = "complete"
    summarized = "summarized"
    simple = "simple"
