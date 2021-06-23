# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Low level parsing workflow with `RecordCollection`

### Changed
- **Breaking:** Parameter `size` and `presize` changed from order (height, width) to (width, height)

## [0.8.0]
Supports pytorch 1.8
### Added
- `iou_thresholds` parameter to `COCOMetric`
- `SimpleConfusionMatrix` Metric
- Negative samples support for mmdetection object detection models

### Changed
- **Breaking:** Albumentations `aug_tfms` defaults.
  - rotate_limit changed from 45 to 15
  - rgb_shift_limit changed from 20 to 10
  - VOC parser uses image sizes from annotation file instead of image
  - bumps fastai to latest version (<2.4)

## [0.7.0]
**BREAKING:** API Refactor

### Added
- Metrics for mmdetection models

### Changed
- **Breaking:** Renamed tasks `default,detect,classif` to `common,detection,classification`
- **Breaking:** Renamed `imageid` to `record_id`
- **Breaking:** Added parameter `is_new` to `Parser.parse_fields`
- Removed all dependencies on `cv2` for visualisation
- Use new composite API for visualisation - covers user defined task names & multiple tasks
- Added a ton of visualisation goodies to `icevision.visualize.draw_data.draw_sample` - user can now
  - use custom fonts
  - control mask thickness
  - control mask blending
  - prettify labels -- show confidence score & capitalise label
  - plot specific and/or exclude specific labels
  - pass in a dictionary mapping labels to specific colors
  - control label height & width padding from bbox edge
  - add border around label for legibility (color is a parameter)

**Breaking:**: Rename `labels->label_ids`, `labels_names->labels` in `LabelsRecordComponent`
- Renamed torchvision resnet backbones:
  - resnet_fpn.resnet18 -> resnet18_fpn
  - resnest_fpn.resnest18 -> resnest18_fpn

**Breaking:** Added parameters `sample` and `keep_image` to `convert_raw_prediction`
**Breaking:** Renamed `VocXmlParser` to `VOCBBoxParser` and `VocMaskParser` to `VOCMaskParser`
**Breaking:** Renamed `predict_dl` to `predict_from_dl`


## [0.6.0b1]
### Added
- mmdetection models

### Changed
- **Breaking:** All `Parser` subclasses need to call `super.__init__`
- **Breaking:** `LabelsMixin.labels` now needs to return `List[Hashable]` instead of `List[int]` (labels names instead of label ids)
- **Breaking:** Model namespace changes e.g. `faster_rcnn` -> `models.torchvision.faster_rcnn`, `efficientdet` -> `models.ross.efficientdet`
- **Breaking:** Renamed `ClassMap.get_name/get_id` to `ClassMap.get_by_name/get_by_id`
- **Breaking:** Removes `idmap` argument from `Parser.parse`. Instead pass `idmap` to the constructor (`__init__`).
- ClassMap is not created inside of the parser, it's not required to instantiate it before
- class_map labels get automatically filled while parsing
- background for class_map is now always 0 (unless no background)
- adds `class_map` to `Record`

### Deleted

## [0.5.2]

### Added
- `aggregate_records_objects` function

### Changed
- Added `label_field` to VIA parser to allow for alternate `region_attribute` names
 
## [0.5.0]

### Added
- Keypoints full support: data API, model and training
- VGG Image Annotator v2 JSON format parser for bboxes
- `figsize` parameter to `show_record` and `show_sample`

### Changed
- improved visualisation for small bboxes
- `COCOMetric` now returns all metrics from pycocotools
- makes torchvision models torchscriptable

## [0.4.0]

### Added
- retinanet: model, dataloaders, predict, ...

### Changed
- **Breaking:** models/rcnn renamed to models/torchvision_models
- tests/models/rcnn renamed to tests/models/torchvision_models

## [0.3.0]

### Added
- pytorch 1.7 support, all dependencies updated
- tutorial with hard negative samples
- ability to skip record while parsing

### Changed
- show_preds visual improvement


## [0.2.2]

### Added
- Cache records after parsing with the new parameter `cache_filepath` added to `Parser.parse` (#504)
- Added `pretrained: bool = True` argument to both faster_rcnn and mask_rcnn `model()` methods. (#516)
- new class `EncodedRLEs`
- all masks get converted to `EncodedRLEs` at parsing time

### Changed
- Removed warning on autofixing masks
- RLE default counts is now COCO style
- renamed `Mask.to_erle` to `Mask.to_erles`

## [0.2.1]

### Changed
- updated matplotlib and ipykernel minimum version for colab compatibility

## [0.2.0]

### IMPORTANT
- **Switched from poetry to setuptools**

### Added
- Function `wandb_img_preds` to help logging bboxes to wandb
- wandb as a soft dependency
- Template code for `parsers.SizeMixin` if `parsers.FilepathMixin` is used
- Get image size without opening image with `get_image_size`
- Ability to skip record while parsing with `AbortParseRecord`
- Autofix for record: `autofix_records` function and `autofix:bool` parameter added to `Parser.parse`
- Record class and mixins, `create_mixed_record` function to help creating Records
- InvalidDataError for BBox
- Catches InvalidDataError while parsing data

### Changed
- **Breaking:** Unified `parsers.SizeMixin` functions `image_width` and `image_height` into a single function `image_width_height`
- Rename Parser `SizeMixin` fields from `width` `height` to `image_width` `image_height`

### Deleted
- Removed `CombinedParser`, all parsing can be done with the standard `Parser`

## [0.1.6]

### Added
- Efficientdet now support empty annotations

### Changed
- Returns float instead of dict on `FastaiMetricAdapter`

## [0.1.5]

### Changed

- Updates fastai2 to the final release version


## [0.1.4]

### Added

- soft import `icedata` in `icevision.all`
- `show_pbar` parameter to `COCOMetric`

### Changed

### Deleted

## [0.1.3]

### Changed
- Effdet as direct dependency

## [0.1.2]

### Added
- `show_results` function for each model

### Changed
- Default `data_splitter` for Parser changed to `RandomSplitter`
- Renamed package from `mantisshrimp` to `icevision`

### Deleted
- Removed `datasets` module to instead use the new `icedata` package

## [0.0.9]

### Added

- `batch, samples = <model_name>.build_infer_batch(dataset)`  
- `preds = <model_name>.predict(model, batch)` 
- `infer_dl = <model_name>.infer_dataloader(dataset)`
- `samples, preds = predict_dl(model, infer_dl)`
- `Dataset.from_images` Contructs a `Dataset` from a list of images (numpy arrays)
- `tfms.A.aug_tfms` for easy access to common augmentation transforms with albumentations
- `tfms.A.resize_and_pad`, useful as a validation transform
- `**predict_kwargs` to `predict_dl` signature
- `from mantisshrimp.all import *` to import internal modules and external imports
- `show` parameter to `show_img`
- `download_gdrive` and `download_and_extract_gdrive`
- New datasets `pennfundan` and `birds`

### Changed

- Renames `AlbuTransform` to `AlbumentationTransforms`
- All `build_batch` method now returns `batch, samples`, the batch is always a tuple of inputs to the model
- `batch_tfms` moved to `tfms.batch`
- `AlbumentationTransforms` moved to `tfms.A.Adapter`
- All parsers function were moved to their own namespace `parsers` instead of being on the global namespace
so, for example, instead of `Parser` now we have to do `parsers.Parser`
- Removed `Parser` word from Mixins, e.g. `ImageidParserMixin` -> `parsers.ImageidMixin`
- Removed `Parser` word from parser default bundle, e.g. `FasterRCNNParser` -> `parsers.FasterRCNN`
- COCO and VOC parsers moved from `datasets` to `parsers`
- `DataSplitter`s moved from `parsers/splits.py` to `utils/data_splitter.py`
- Renames `*_dataloader` to `*_dl`, e.g. `mask_rcnn.train_dataloader` to `mask_rcnn.train_dl`
- Moves `RecordType` from `parsers` to `core`
- Refactors `IDMap`, adds methods `get_name` and `get_id`
- Moves `IDMap` from `utils` to `data`
- `DataSplitter.split` now receives `idmap` instead of `ids`


## [0.0.0-pre-release]

### Added

- `CaptureStdout` for capturing writes to stdout (print), e.g. from COCOMetric
- `mantisshrimp.models.<model_name>.convert_raw_predictions` to convert raw preds (tensors output from the model) to library standard dict
- `COCOMetricType` for selecting what metric type to use (`bbox`, `mask`, `keypoints`)
- `COCOMetric` fixed
- `sort` parameter for `get_image_files`
- `ClassMap`: A class that handles the mapping between ids and names, with the optional insertion of the background class

### Changed

-  All dataloaders now return the batch and the records, e.g. `return (images, targets), records`
- `Metric.accumulate` signature changed to `(records, preds)`, reflects in `FastaiMetricAdapter` and `LightningModelAdapter`
- `datasets.<name>.CLASSES` substituted by a function `datasets.<name>.class_map` that returns a `ClassMap`
- `datasets.voc.VocXmlParser`, `show` methods: parameter `classes: Sequence[str]` substituted by `class_map: ClassMap`
- `datasets.fridge.parser`, `datasets.pets.parser`: additional required parameter `class_map`

### Removed
- `MantisFasterRCNN`, `MantisMaskRCNN`
- `MantisEfficientDet`
- `CategoryMap`, `Category`
- `MantisModule`


## Links  
[Unreleased]: https://github.com/airctic/mantisshrimp/tree/master

