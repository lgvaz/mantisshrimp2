from icevision.models.utils import *
from icevision.models.interpretation import *

from icevision.models import torchvision

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies, _SoftDependencies

if SoftDependencies.effdet:
    # backwards compatibility
    from icevision.models.ross import efficientdet
    from icevision.models import ross

if SoftDependencies.mmdet:
    from icevision.models import mmdet

if SoftDependencies.yolov5:
    # HACK: yolov5 changes matplotlib backend here: https://github.com/ultralytics/yolov5/blob/77415a42e5975ea356393c9f1d5cff0ae8acae2c/utils/plots.py#L26
    import matplotlib

    backend = matplotlib.get_backend()
    from icevision.models import ultralytics

    matplotlib.use(backend)
    matplotlib.rcdefaults()

if SoftDependencies.fastai:
    from icevision.models import fastai
