__all__ = [
    "coco",
    "COCOBaseParser",
    "COCOBBoxParser",
    "COCOMaskParser",
    "COCOKeyPointsParser",
    "COCOKeypointsMetadata",
]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


# TODO: Deprecated
def coco(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    mask: bool = True,
    idmap: Optional[IDMap] = None,
) -> Parser:
    logger.warning(
        "This function will be deprecated, instantiate the concrete "
        "classes instead: `COCOBBoxParser`, `COCOMaskParser`, `COCOKeypointsParser`"
    )
    parser_cls = COCOMaskParser if mask else COCOBBoxParser
    return parser_cls(annotations_file, img_dir, idmap=idmap)


class COCOBaseParser(Parser):
    def __init__(
        self,
        annotations_filepath: Union[str, Path],
        img_dir: Union[str, Path],
        idmap: Optional[IDMap] = None,
    ):

        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)

        self._record_id2info = {o["id"]: o for o in self.annotations_dict["images"]}

        categories = self.annotations_dict["categories"]
        id2class = {o["id"]: o["name"] for o in categories}
        id2class[0] = BACKGROUND
        # coco has non sequential ids, we fill the blanks with `None`, check #668 for more info
        classes = [None for _ in range(max(id2class.keys()) + 1)]
        for i, name in id2class.items():
            classes[i] = name
        self.class_map = ClassMap(classes)

        super().__init__(template_record=self.template_record(), idmap=idmap)

    def __iter__(self):
        yield from self.annotations_dict["annotations"]

    def __len__(self):
        return len(self.annotations_dict["annotations"])

    def template_record(self) -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                AreasRecordComponent(),
                IsCrowdsRecordComponent(),
            )
        )

    def prepare(self, o):
        self._info = self._record_id2info[o["image_id"]]

    def record_id(self, o) -> int:
        return o["image_id"]

    def filepath(self, o) -> Path:
        return self.img_dir / self._info["file_name"]

    def img_size(self, o) -> ImgSize:
        return get_img_size(self.filepath(o))

    def labels_ids(self, o) -> List[Hashable]:
        return [o["category_id"]]

    def areas(self, o) -> List[float]:
        return [o["area"]]

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]]

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.img_size(o))

        # TODO: is class_map still a issue here?
        record.detection.set_class_map(self.class_map)
        record.detection.add_labels_by_id(self.labels_ids(o))
        record.detection.add_areas(self.areas(o))
        record.detection.add_iscrowds(self.iscrowds(o))


class COCOBBoxParser(COCOBaseParser):
    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])]

    def template_record(self) -> BaseRecord:
        record = super().template_record()
        record.add_component(BBoxesRecordComponent())
        return record

    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new=is_new)
        record.detection.add_bboxes(self.bboxes(o))


class COCOMaskParser(COCOBBoxParser):
    def masks(self, o) -> List[MaskArray]:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return [RLE.from_coco(seg["counts"])]
        else:
            return [Polygon(seg)]

    def template_record(self) -> BaseRecord:
        record = super().template_record()
        record.add_component(InstanceMasksRecordComponent())
        return record

    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new=is_new)
        record.detection.add_masks(self.masks(o))


class COCOKeyPointsParser(COCOBBoxParser):
    def template_record(self) -> BaseRecord:
        record = super().template_record()
        record.add_component(KeyPointsRecordComponent())
        return record

    def keypoints(self, o) -> List[KeyPoints]:
        return (
            [KeyPoints.from_xyv(o["keypoints"], COCOKeypointsMetadata)]
            if sum(o["keypoints"]) > 0
            else []
        )

    def labels_ids(self, o) -> List[Hashable]:
        if sum(o["keypoints"]) <= 0:
            return []
        return super().labels_ids(o)

    def areas(self, o) -> List[float]:
        return [o["area"]] if sum(o["keypoints"]) > 0 else []

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]] if sum(o["keypoints"]) > 0 else []

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])] if sum(o["keypoints"]) > 0 else []

    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new=is_new)
        record.detection.add_keypoints(self.keypoints(o))


class COCOConnectionsColor:
    head = (220, 30, 200)
    torso = (100, 240, 100)
    right_arm = (170, 170, 100)
    left_arm = (120, 120, 230)
    left_leg = (230, 120, 130)
    right_leg = (230, 180, 190)


class COCOKeypointsMetadata(KeypointsMetadata):
    labels = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    connections = (
        KeypointConnection(0, 1, COCOConnectionsColor.head),
        KeypointConnection(0, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 3, COCOConnectionsColor.head),
        KeypointConnection(2, 4, COCOConnectionsColor.head),
        KeypointConnection(3, 5, COCOConnectionsColor.head),
        KeypointConnection(4, 6, COCOConnectionsColor.head),
        KeypointConnection(5, 6, COCOConnectionsColor.torso),
        KeypointConnection(5, 7, COCOConnectionsColor.left_arm),
        KeypointConnection(5, 11, COCOConnectionsColor.torso),
        KeypointConnection(6, 8, COCOConnectionsColor.right_arm),
        KeypointConnection(6, 12, COCOConnectionsColor.torso),
        KeypointConnection(7, 9, COCOConnectionsColor.left_arm),
        KeypointConnection(8, 10, COCOConnectionsColor.right_arm),
        KeypointConnection(11, 12, COCOConnectionsColor.torso),
        KeypointConnection(13, 11, COCOConnectionsColor.left_leg),
        KeypointConnection(14, 12, COCOConnectionsColor.right_leg),
        KeypointConnection(15, 13, COCOConnectionsColor.left_leg),
        KeypointConnection(16, 14, COCOConnectionsColor.right_leg),
    )
