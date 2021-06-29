import pytest
from icevision import *
from icevision.imports import *
from icevision.models.torchvision import faster_rcnn, keypoint_rcnn
from icevision.models.ross import efficientdet
from icevision.models.ross.efficientdet.backbones import *
import albumentations as A


@pytest.fixture(scope="session")
def samples_source():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="session")
def coco_record_id_map():
    return IDMap()


@pytest.fixture()
def fridge_efficientdet_model() -> nn.Module:
    WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m2/fridge_tf_efficientdet_lite0.pt"
    # TODO: HACK 5+1 in num_classes (becaues of change in model.py)
    backbone = models.ross.efficientdet.backbones.tf_lite0(pretrained=False)
    model = efficientdet.model(backbone=backbone, num_classes=5, img_size=384)

    state_dict = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    return model


@pytest.fixture()
def fridge_faster_rcnn_model() -> nn.Module:
    backbone = models.torchvision.faster_rcnn.backbones.resnet18_fpn(pretrained=False)
    return faster_rcnn.model(num_classes=5, backbone=backbone)


@pytest.fixture()
def camvid_class_map(samples_source) -> ClassMap:
    codes = list(np.loadtxt(samples_source / "camvid/codes.txt", dtype=str))
    return ClassMap(codes, background=None)


@pytest.fixture()
def camvid_records(samples_source, camvid_class_map) -> RecordCollection:
    images_dir = samples_source / "camvid/images"
    labels_dir = samples_source / "camvid/labels"
    image_files = get_image_files(images_dir)

    records = RecordCollection(SemanticSegmentationRecord)

    for image_file in pbar(image_files):
        record = records.get_by_record_id(image_file.stem)

        if record.is_new:
            record.set_filepath(image_file)
            record.set_img_size(get_img_size(image_file))
            record.segmentation.set_class_map(camvid_class_map)

        mask_file = SemanticMaskFile(labels_dir / f"{image_file.stem}_P.png")
        record.segmentation.set_mask(mask_file)

    records = records.autofix()
    # list of 5 records
    return records


@pytest.fixture()
def camvid_ds(camvid_records) -> Tuple[Dataset, Dataset]:

    train_records, valid_records = camvid_records.make_splits(
        RandomSplitter([0.8, 0.2], seed=0)
    )

    IMG_SIZE = 64
    tfms_ = tfms.A.Adapter([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize()])

    train_ds = Dataset(train_records, tfms_)
    valid_ds = Dataset(valid_records, tfms_)
    assert len(train_ds) == 4
    assert len(valid_ds) == 1

    return train_ds, valid_ds


@pytest.fixture(scope="module")
def fridge_ds(samples_source, fridge_class_map) -> Tuple[Dataset, Dataset]:
    IMG_SIZE = 384

    parser = parsers.VOCBBoxParser(
        annotations_dir=samples_source / "fridge/odFridgeObjects/annotations",
        images_dir=samples_source / "fridge/odFridgeObjects/images",
        class_map=fridge_class_map,
    )

    data_splitter = RandomSplitter([0.5, 0.5], seed=42)
    train_records, valid_records = parser.parse(data_splitter)

    tfms_ = tfms.A.Adapter([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize()])

    train_ds = Dataset(train_records, tfms_)
    valid_ds = Dataset(valid_records, tfms_)

    return train_ds, valid_ds


@pytest.fixture(params=[2, 3])
def fridge_efficientdet_dls(fridge_ds, request) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = fridge_ds
    train_dl = efficientdet.train_dl(train_ds, batch_size=request.param)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=request.param)

    return train_dl, valid_dl


@pytest.fixture(params=[2, 3])
def fridge_faster_rcnn_dls(fridge_ds, request) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = fridge_ds
    train_dl = faster_rcnn.train_dl(train_ds, batch_size=request.param)
    valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=request.param)

    return train_dl, valid_dl


@pytest.fixture(scope="session")
def voc_class_map():
    classes = sorted(
        {
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        }
    )

    return ClassMap(classes=classes)


@pytest.fixture(scope="session")
def fridge_class_map():
    classes = sorted({"milk_bottle", "carton", "can", "water_bottle"})
    return ClassMap(classes)


# COCO fixtures
@pytest.fixture(scope="session")
def coco_dir():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="module")
def coco_bbox_parser(coco_dir):
    return parsers.COCOBBoxParser(
        annotations_filepath=coco_dir / "annotations.json",
        img_dir=coco_dir / "images",
    )


@pytest.fixture(scope="module")
def coco_mask_parser(coco_dir, coco_record_id_map):
    return parsers.COCOMaskParser(
        annotations_filepath=coco_dir / "annotations.json",
        img_dir=coco_dir / "images",
        idmap=coco_record_id_map,
    )


@pytest.fixture(scope="module")
def coco_keypoints_parser(coco_dir):
    return parsers.COCOKeyPointsParser(
        annotations_filepath=coco_dir / "keypoints_annotations.json",
        img_dir=coco_dir / "images",
    )


@pytest.fixture(scope="module")
def coco_mask_records(coco_mask_parser):
    return coco_mask_parser.parse(data_splitter=SingleSplitSplitter())[0]


@pytest.fixture
def coco_record(coco_mask_records):
    return deepcopy(coco_mask_records[0])


@pytest.fixture
def coco_sample(coco_record):
    return deepcopy(coco_record.load())


@pytest.fixture
def keypoints_img_128372():
    return [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        406,
        218,
        1,
        423,
        218,
        1,
        394,
        235,
        2,
        431,
        236,
        2,
        384,
        257,
        2,
        437,
        269,
        2,
        396,
        281,
        2,
        440,
        292,
        2,
        402,
        292,
        2,
        423,
        293,
        2,
        401,
        330,
        2,
        423,
        329,
        2,
        399,
        370,
        2,
        421,
        371,
        2,
    ]


# VIA fixtures
@pytest.fixture(scope="session")
def via_bbox_class_map():
    classes = sorted({"z", "c", "n", "o"})

    return ClassMap(classes=classes)


@pytest.fixture(scope="session")
def via_dir():
    return Path(__file__).absolute().parent.parent / "samples/via"


# OCHumanKeypoints fixtures
class OCHumanKeypointsMetadata(KeypointsMetadata):
    labels = (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "head",
        "neck",
        "right_ear",
        "left_ear",
        "nose",
        "right_eye",
        "left_eye",
    )


@pytest.fixture(scope="module")
def ochuman_ds(samples_source) -> Tuple[Dataset, Dataset]:
    class OCHumanParser(Parser):
        def __init__(self, annotations_filepath, img_dir):
            self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
            self.img_dir = Path(img_dir)
            self.class_map = ClassMap(["person"])
            super().__init__(template_record=self.template_record())

        def __iter__(self):
            yield from self.annotations_dict["images"]

        def __len__(self):
            return len(self.annotations_dict["images"])

        def template_record(self):
            return BaseRecord(
                (
                    FilepathRecordComponent(),
                    KeyPointsRecordComponent(),
                    InstancesLabelsRecordComponent(),
                    BBoxesRecordComponent(),
                )
            )

        def record_id(self, o):
            return int(o["image_id"])

        def parse_fields(self, o, record: BaseRecord, is_new) -> None:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.img_size(o))

            record.detection.set_class_map(self.class_map)
            record.detection.add_labels(self.labels(o))
            record.detection.add_bboxes(self.bboxes(o))
            record.detection.add_keypoints(self.keypoints(o))

        def filepath(self, o):
            return self.img_dir / o["file_name"]

        def keypoints(self, o):
            return [
                KeyPoints.from_xyv(kps["keypoints"], OCHumanKeypointsMetadata)
                for kps in o["annotations"]
                if kps["keypoints"] is not None
            ]

        def img_size(self, o) -> Tuple[int, int]:
            return get_img_size(self.filepath(o))

        def labels(self, o) -> List[Hashable]:
            return [
                "person" for ann in o["annotations"] if ann["keypoints"] is not None
            ]

        def bboxes(self, o) -> List[BBox]:
            return [
                BBox.from_xyxy(*ann["bbox"])
                for ann in o["annotations"]
                if ann["keypoints"] is not None
            ]

    parser = OCHumanParser(
        samples_source / "ochuman/annotations/ochuman.json",
        samples_source / "ochuman/images/",
    )
    train_records, valid_records = parser.parse(
        data_splitter=RandomSplitter([0.8, 0.2])
    )

    presize = 64
    size = 32

    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize, crop_fn=None), tfms.A.Normalize()]
    )

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    return train_ds, valid_ds


@pytest.fixture()
def ochuman_keypoints_dls(ochuman_ds) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = ochuman_ds
    train_dl = keypoint_rcnn.train_dl(train_ds, batch_size=2)
    valid_dl = keypoint_rcnn.valid_dl(valid_ds, batch_size=2)

    return train_dl, valid_dl


### New conftest ###
@pytest.fixture
def object_detection_record(samples_source):
    record = ObjectDetectionRecord()

    record.set_record_id(1)
    record.set_filepath(samples_source / "voc/JPEGImages/2007_000063.jpg")
    record.set_img_size(ImgSize(width=500, height=375))
    record.detection.set_class_map(ClassMap(["a", "b"]))
    record.detection.add_labels_by_id([1, 2])
    record.detection.add_bboxes(
        [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(10, 20, 30, 40)]
    )

    return record


@pytest.fixture
def instance_segmentation_record(object_detection_record):
    record = object_detection_record
    record.add_component(InstanceMasksRecordComponent())

    record.detection.add_masks([MaskArray(np.ones((2, 4, 4), dtype=np.uint8))])

    return record


@pytest.fixture
def empty_annotations_record():
    record = BaseRecord(
        (
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            InstanceMasksRecordComponent(),
        )
    )

    img = 255 * np.ones((4, 4, 3), dtype=np.uint8)
    record.set_record_id(1)
    record.set_img(img)

    return record


@pytest.fixture
def infer_dataset(samples_source):
    img = open_img(samples_source / "voc/JPEGImages/2007_000063.jpg")
    return Dataset.from_images([img] * 2)


component_field = {
    RecordIDRecordComponent: "record_id",
    ClassMapRecordComponent: "class_map",
    FilepathRecordComponent: "img",
    ImageRecordComponent: "img",
    SizeRecordComponent: "img_size",
    InstancesLabelsRecordComponent: "labels",
    BBoxesRecordComponent: "bboxes",
    InstanceMasksRecordComponent: "masks",
    KeyPointsRecordComponent: "keypoints",
    AreasRecordComponent: "areas",
    IsCrowdsRecordComponent: "iscrowds",
}


@pytest.fixture
def check_attributes_on_component():
    def _inner(record):
        for component in record.components:
            name = component_field[component.__class__]
            task_subfield = getattr(record, component.task.name)
            assert getattr(task_subfield, name) is getattr(component, name)

    return _inner
