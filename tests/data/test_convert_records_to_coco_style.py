import pytest
import icevision
from icevision.all import *


@pytest.fixture()
def coco_records(coco_mask_records):
    return convert_records_to_coco_style(coco_mask_records)


@pytest.fixture()
def expected_records():
    source = Path(icevision.__file__).parent.parent / "samples"
    dataset = torchvision.datasets.CocoDetection(source, source / "annotations.json")
    return dataset.coco.dataset


def test_convert_records_to_coco_style_images(
    coco_records, expected_records, coco_record_id_map
):
    # remove id 89 because it doesn't contain any annotations and it will be removed by the parser
    # also remove some unnecessary fields like flickr_url, coco_url, ...
    expected_images = [
        {name: o[name] for name in ["id", "file_name", "height", "width"]}
        for o in expected_records["images"]
        if o["id"] != 89
    ]
    # some of the checks need the lists to be in the same order
    expected_images = sorted(expected_images, key=itemgetter("id"))
    images = sorted(coco_records["images"], key=itemgetter("id"))

    assert len(images) == len(expected_images)
    assert images[0].keys() == expected_images[0].keys()
    assert images == expected_images


def test_convert_records_to_coco_style_annotations(
    coco_mask_records, coco_records, expected_records, coco_record_id_map
):
    annotations = coco_records["annotations"]
    expected_annotations = expected_records["annotations"]

    assert len(expected_annotations) == len(annotations)
    assert annotations[0].keys() == expected_annotations[0].keys()

    ## check if two unordered lists of dicts are approximate the same ##

    # the annotations ids don't matter and don't match
    for i in range(len(annotations)):
        del annotations[i]["id"]
        del expected_annotations[i]["id"]

    # use record_id_map to convert to ids used by the parser
    # for annotation in expected_annotations:
    #     annotation["image_id"] = coco_record_id_map[annotation["image_id"]]

    # # convert label ids
    # original_cocoid2label = {o["id"]: o["name"] for o in expected_records["categories"]}
    # class_map = coco_mask_records[0].detection.class_map

    # sort items so we compare each annotation with it's corresponded pair
    sorted_annotations = sorted(sorted(d.items()) for d in annotations)
    sorted_expectation = sorted(sorted(d.items()) for d in expected_annotations)

    # check each (key, value) individually (so we can use almost_equal)
    for annotation, expected_annotation in zip(sorted_annotations, sorted_expectation):
        # remember that each annotation is now in the form: (key, value)
        for (k1, v1), (k2, v2) in zip(annotation, expected_annotation):
            assert k1[0] == k2[0]
            if k1 == "segmentation":
                # TODO: Skipping segmentation check
                pass
            # elif k1 == "category_id":
            #     assert class_map.get_by_id(v1) == original_cocoid2label[v2]
            else:
                np.testing.assert_almost_equal(v1, v2)


def test_coco_api_from_records(coco_mask_records):
    coco_api = coco_api_from_records(coco_mask_records)
    image_info = coco_api.loadImgs([343934])
    expected = [
        {"id": 343934, "file_name": "000000343934.jpg", "width": 640, "height": 480}
    ]
    assert image_info == expected


@pytest.mark.skip
def test_convert_record_to_coco_annotations_empty():
    record = {
        "labels": [],
        "bboxes": [],
        "masks": [],
        "scores": [],
        "masks": EncodedRLEs(),
    }
    res = convert_record_to_coco_annotations(record)
    assert set(res.keys()) == {
        "image_id",
        "category_id",
        "bbox",
        "area",
        "iscrowd",
        "score",
        "segmentation",
    }
