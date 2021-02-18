# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:35:32 2021

@author: Kazu
"""
from enum import Enum


class PossibleDtypes(str, Enum):
    image = "Image"
    fasttext = "FastText"
    unsupervised = "Unsupervised"
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
