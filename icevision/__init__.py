__version__ = "0.2.0b2"

from icevision.utils import *
from icevision.core import *
from icevision import parsers
from icevision import tfms
from icevision.data import *
from icevision import backbones
from icevision import models
from icevision.models import *
from icevision.metrics import *
from icevision.visualize import *

import icevision.test_utils as test_utils

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
