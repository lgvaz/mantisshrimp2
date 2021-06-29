__all__ = [
    "create_coco_api",
    "convert_records_to_coco_style",
    "convert_preds_to_coco_style",
    "convert_record_to_coco_annotations",
    "coco_api_from_records",
    "coco_api_from_preds",
    "create_coco_eval",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def create_coco_api(coco_records) -> COCO:
    """Create COCO dataset api

    Args:
        coco_records: Records in coco style (use convert_records_to_coco_style to convert
        records to coco style.
    """
    coco_ds = COCO()
    coco_ds.dataset = coco_records
    coco_ds.createIndex()

    return coco_ds


def coco_api_from_preds(preds, show_pbar: bool = False) -> COCO:
    coco_preds = convert_preds_to_coco_style(preds, show_pbar=show_pbar)
    return create_coco_api(coco_preds)


def coco_api_from_records(records, show_pbar: bool = False) -> COCO:
    """Create pycocotools COCO dataset from records"""
    coco_records = convert_records_to_coco_style(records, show_pbar=show_pbar)
    return create_coco_api(coco_records=coco_records)


def create_coco_eval(
    records,
    preds,
    metric_type: str,
    iou_thresholds: Optional[Sequence[float]] = None,
    show_pbar: bool = False,
) -> COCOeval:
    assert len(records) == len(preds)

    for record, pred in zip(records, preds):
        pred.record_id = record.record_id
        pred.height = record.height
        pred.width = record.width
        # needs 'filepath' for mask `coco.py#418`
        pred.filepath = record.filepath

    target_ds = coco_api_from_records(records, show_pbar=show_pbar)
    pred_ds = coco_api_from_preds(preds, show_pbar=show_pbar)

    coco_eval = COCOeval(target_ds, pred_ds, metric_type)
    if iou_thresholds is not None:
        coco_eval.iouThrs = iou_thresholds

    return coco_eval


def convert_record_to_coco_image(record) -> dict:
    image = {}
    image["id"] = record.record_id
    image["file_name"] = Path(record.filepath).name
    image["width"] = record.width
    image["height"] = record.height
    return image


def convert_record_to_coco_annotations(record):
    annotations_dict = {
        "image_id": [],
        "category_id": [],
        "bbox": [],
        "area": [],
        "iscrowd": [],
    }
    # build annotations field
    for label in record.detection.label_ids:
        annotations_dict["image_id"].append(record.record_id)
        annotations_dict["category_id"].append(label)

    for bbox in record.detection.bboxes:
        annotations_dict["bbox"].append(list(bbox.xywh))

    if hasattr(record.detection, "areas"):
        for area in record.detection.areas:
            annotations_dict["area"].append(area)
    else:
        for bbox in record.detection.bboxes:
            annotations_dict["area"].append(bbox.area)

    # HACK: Because of prepare_record, mask should always be `MaskArray`,
    # maybe the for loop is not required?
    if hasattr(record.detection, "masks"):
        # HACK: Hacky again!
        mask_array = record.detection.mask_array
        if mask_array is None:
            mask_array = MaskArray.from_masks(
                record.detection.masks, record.height, record.width
            )

        annotations_dict["segmentation"] = mask_array.to_erles(
            record.height, record.width
        ).erles

        # if isinstance(masks, MaskArray):
        #     masks = masks.to_erles(record.height, record.width)

        # if isinstance(masks, EncodedRLEs):
        #     annotations_dict["segmentation"] = masks.erles

        # else:
        #     raise RuntimeError(
        #         "masks are expected to be EncodedRLEs only, "
        #         "if you get this error please open an issue on github."
        #     )
        # annotations_dict["segmentation"] = []
        # for mask in record.detection.masks:
        #     if isinstance(mask, MaskArray):
        #         # HACK: see previous hack
        #         assert len(mask.shape) == 2
        #         mask2 = MaskArray(mask.data[None])
        #         rles = mask2.to_coco_rle(record.height, record.width)
        #         annotations_dict["segmentation"].extend(rles)
        #     elif isinstance(mask, Polygon):
        #         annotations_dict["segmentation"].append(mask.points)
        #     elif isinstance(mask, RLE):
        #         coco_rle = {
        #             "counts": mask.to_coco(),
        #             "size": [record.height, record.width],
        #         }
        #         annotations_dict["segmentation"].append(coco_rle)
        #     elif isinstance(mask, MaskFile):
        #         rles = mask.to_coco_rle(record.height, record.width)
        #         annotations_dict["segmentation"].extend(rles)
        #     elif isinstance(mask, EncodedRLEs):
        #         annotations_dict["segmentation"].append(mask.erles)
        #     else:
        #         msg = f"Mask type {type(mask)} unsupported"
        #         raise ValueError(msg)

    # TODO: is auto assigning a value for iscrowds dangerous (may hurt the metric value?)
    if not hasattr(record.detection, "iscrowds"):
        record.detection.iscrowds = [0] * len(record.detection.label_ids)
    for iscrowd in record.detection.iscrowds:
        annotations_dict["iscrowd"].append(iscrowd)

    if hasattr(record.detection, "scores"):
        annotations_dict["score"] = record.detection.scores

    return annotations_dict


def convert_preds_to_coco_style(preds, show_pbar: bool = False):
    return convert_records_to_coco_style(
        records=preds, images=True, categories=False, show_pbar=show_pbar
    )


def convert_records_to_coco_style(
    records,
    images: bool = True,
    annotations: bool = True,
    categories: bool = True,
    show_pbar: bool = True,
):
    """Converts records from library format to coco format.
    Inspired from: https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py#L146
    """
    images_ = []
    annotations_dict = defaultdict(list)

    for record in pbar(records, show=show_pbar):
        if images:
            image_ = convert_record_to_coco_image(record)
            images_.append(image_)

        if annotations:
            annotations_ = convert_record_to_coco_annotations(record)
            for k, v in annotations_.items():
                annotations_dict[k].extend(v)

    if annotations:
        annotations_dict = dict(annotations_dict)
        if not allequal([len(o) for o in annotations_dict.values()]):
            raise RuntimeError("Mismatch lenght of elements")

        # convert dict of lists to list of dicts
        annotations_ = []
        for i in range(len(annotations_dict["image_id"])):
            annotation = {k: v[i] for k, v in annotations_dict.items()}
            # annotations should be initialized starting at 1 (torchvision issue #1530)
            annotation["id"] = i + 1
            annotations_.append(annotation)

    if categories:
        categories_set = set(annotations_dict["category_id"])
        categories_ = [{"id": i} for i in categories_set]

    res = {}
    if images_:
        res["images"] = images_
    if annotations:
        res["annotations"] = annotations_
    if categories:
        res["categories"] = categories_

    return res
