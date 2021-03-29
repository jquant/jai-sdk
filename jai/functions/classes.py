from enum import Enum

__all__ = ['FieldName', 'Mode', 'PossibleDtypes']


class PossibleDtypes(str, Enum):
    image = "Image"
    fasttext = "FastText"
    selfsupervised = "SelfSupervised"
    supervised = "Supervised"
    text = "Text"
    edit = "TextEdit"


class FieldName(str, Enum):
    text = "text"
    image = "image_base64"

    def __str__(self):
        return str(self.value)


class Mode(str, Enum):
    complete = "complete"
    summarized = "summarized"
    simple = "simple"
