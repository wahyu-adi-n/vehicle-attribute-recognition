# Setting Up the Labels
from config.configs import *


def decode_label(index):
    return LABELS[index]


def encode_label_from_path(path):
    for index, value in enumerate(LABELS):
        if value in path:
            return index
