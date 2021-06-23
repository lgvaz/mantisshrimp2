__all__ = [
    "ObjectDetectionRecord",
    "InstanceSegmentationRecord",
    "SemanticSegmentationRecord",
    "KeypointsRecord",
]

from icevision.core.record import *
from icevision.core.record_components import *
from icevision.core import tasks


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def InstanceSegmentationRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            MasksRecordComponent(),
        )
    )


def SemanticSegmentationRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            ClassMapRecordComponent(task=tasks.segmentation),
            SemanticMasksRecordComponent(),
        )
    )


def KeypointsRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )
