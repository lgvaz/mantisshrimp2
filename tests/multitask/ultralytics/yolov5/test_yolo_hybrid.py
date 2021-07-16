# from numpy.lib.arraysetops import isin
# import pytest
# from icevision.imports import *
# from icevision.models.multitask.ultralytics.yolov5.yolo_hybrid import *
# from icevision.models.multitask.utils import *


# @pytest.fixture
# def model():
#     return HybridYOLOV5(
#         cfg="models/yolov5m.yaml",
#         classifier_configs=dict(
#             framing=ClassifierConfig(out_classes=10, num_fpn_features=10),
#             saturation=ClassifierConfig(out_classes=20, num_fpn_features=None),
#         ),
#     )


# def x():
#     return torch.rand(1, 3, 224, 224)


# def test_forward(model, x):
#     det_out, clf_out = model.forward_once(x)
#     assert isinstance(det_out, TensorList)
#     assert isinstance(clf_out, TensorDict)
#     assert det_out[0].ndim == 5


# def test_forward_eval(model, x):
#     det_out, clf_out = model.forward_once(x)

#     assert len(det_out == 2)
#     assert isinstance(det_out[0], Tensor)
#     assert isinstance(det_out[1], TensorList)


# def test_feature_extraction(model, x):
#     det_out, clf_out = model.forward_once(
#         forward_detection=False, forward_classification=False
#     )
#     assert det_out[0].ndim == 3
#     assert clf_out == {}


# def test_fwd_inference(model, x):
#     det_out, clf_out = model.forward_once(activate_classification=True)
#     torch.allclose(clf_out["framing"].sum(), tensor(1.0))
