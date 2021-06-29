__all__ = ["voc", "VOCBBoxParser", "VOCMaskParser"]

import xml.etree.ElementTree as ET
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.parsers.parser import *


def voc(
    annotations_dir: Union[str, Path],
    images_dir: Union[str, Path],
    class_map: Optional[ClassMap] = None,
    masks_dir: Optional[Union[str, Path]] = None,
    idmap: Optional[IDMap] = None,
):
    logger.warning(
        "This function will be deprecated, instantiate the concrete "
        "classes instead: `VOCBBoxParser`, `VOCMaskParser`"
    )
    if not masks_dir:
        return VOCBBoxParser(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            class_map=class_map,
            idmap=idmap,
        )
    else:
        return VOCMaskParser(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            masks_dir=masks_dir,
            class_map=class_map,
            idmap=idmap,
        )


# TODO: Rename to VOCBBoxParser?
class VOCBBoxParser(Parser):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        super().__init__(template_record=self.template_record(), idmap=idmap)
        self.class_map = class_map or ClassMap().unlock()
        self.images_dir = Path(images_dir)

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = get_files(self.annotations_dir, extensions=[".xml"])

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def template_record(self) -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

    def record_id(self, o) -> Hashable:
        return str(Path(self._filename).stem)

    def prepare(self, o):
        tree = ET.parse(str(o))
        self._root = tree.getroot()
        self._filename = self._root.find("filename").text
        self._size = self._root.find("size")

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.img_size(o))

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(self.labels(o))
        record.detection.add_bboxes(self.bboxes(o))

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    def img_size(self, o) -> ImgSize:
        width = int(self._size.find("width").text)
        height = int(self._size.find("height").text)
        return ImgSize(width=width, height=height)

    def labels(self, o) -> List[Hashable]:
        labels = []
        for object in self._root.iter("object"):
            label = object.find("name").text
            labels.append(label)

        return labels

    def bboxes(self, o) -> List[BBox]:
        def to_int(x):
            return int(float(x))

        bboxes = []
        for object in self._root.iter("object"):
            xml_bbox = object.find("bndbox")
            xmin = to_int(xml_bbox.find("xmin").text)
            ymin = to_int(xml_bbox.find("ymin").text)
            xmax = to_int(xml_bbox.find("xmax").text)
            ymax = to_int(xml_bbox.find("ymax").text)

            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

        return bboxes


class VOCMaskParser(VOCBBoxParser):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        super().__init__(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            class_map=class_map,
            idmap=idmap,
        )
        self.masks_dir = masks_dir
        self.mask_files = get_image_files(masks_dir)

        self._record_id2maskfile = {self.record_id_mask(o): o for o in self.mask_files}

        # filter annotations
        masks_ids = frozenset(self._record_id2maskfile.keys())
        self._intersection = []
        for item in super().__iter__():
            super().prepare(item)
            if super().record_id(item) in masks_ids:
                self._intersection.append(item)

    def __len__(self):
        return len(self._intersection)

    def __iter__(self):
        yield from self._intersection

    def template_record(self) -> BaseRecord:
        record = super().template_record()
        record.add_component(InstanceMasksRecordComponent())
        return record

    def record_id_mask(self, o) -> Hashable:
        """Should return the same as `record_id` from parent parser."""
        return str(Path(o).stem)

    def parse_fields(self, o, record, is_new):
        super().parse_fields(o, record, is_new=is_new)
        record.detection.add_masks(self.masks(o))

    def masks(self, o) -> List[Mask]:
        mask_file = self._record_id2maskfile[self.record_id(o)]
        return [VocMaskFile(mask_file)]
