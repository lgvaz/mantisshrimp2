from icevision.imports import *
from icevision.core import *
from icevision.core.tasks import Task
from torch.utils.data import Dataset
from icevision.data.dataset import Dataset as RecordDataset
from icevision.utils.utils import normalize, flatten

import icevision.tfms as tfms
import torchvision.transforms as Tfms

__all__ = ["HybridAugmentationsRecordDataset", "RecordDataset"]


class HybridAugmentationsRecordDataset(Dataset):
    """
    A Dataset that allows you to apply different augmentations to different tasks in your
      record. `detection_transforms` are applied to the `detection` task specifically, and
      `classification_transforms_groups` describe how to group and apply augmentations to
      the classification tasks in the record.

    This object stores the records internally and dynamically attaches an `img` component
      to each task when being fetched. Some basic validation is done on init to ensure that
      the given transforms cover all tasks described in the record.

    Important NOTE: All images are returned as normalised numpy arrays upon fetching. If
      running in `debug` mode, normalisation is skipped and PIL Images are returned inside
      the record instead. This is done to facilitate visual inspection of the transforms
      applied to the images

    Arguments:
        * records: A list of records where only the `common` attribute has an `img`. Upon fetching,
                   _each_ task in the record will have an `img` attribute added to it based on the
                   `classification_transforms_groups`
        * classification_transforms_groups <Dict[str, Dict[str, Union[Tfms.Compose, List[str]]]] : a dict
            that creates groups of tasks, where each task receives the same transforms and gets a dedicated
            forward pass in the network. See below for an example.
        * detection_transforms <tfms.A.Adapter> - Icevision albumentations adapter for detection transforms.
        * norm_mean <List[float]> : norm mean stats
        * norm_std <List[float]> : norm stdev stats
        * debug <bool> : If true, prints info & unnormalised `PIL.Image`s are returned on fetching items

    Usage:
        Sample record:
            BaseRecord

            common:
                - Image ID: 4
                - Filepath: sample_image.png
                - Image: 640x640x3 <np.ndarray> Image
                - Image size ImgSize(width=640, height=640)
            color_saturation:
                - Class Map: <ClassMap: {'desaturated': 0, 'neutral': 1}>
                - Labels: [1]
            shot_composition:
                - Class Map: <ClassMap: {'balanced': 0, 'center': 1}>
                - Labels: [1]
            detection:
                - BBoxes: [<BBox (xmin:29, ymin:91, xmax:564, ymax:625)>]
                - Class Map: <ClassMap: {'background': 0, 'person': 1}>
                - Labels: [1]
            shot_framing:
                - Class Map: <ClassMap: {'01-wide': 0, '02-medium': 1, '03-closeup': 2}>
                - Labels: [3]

        classification_transforms_groups = {
                "group1": dict(
                    tasks=["shot_composition"],
                    transforms=Tfms.Compose([
                        Tfms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                        Tfms.RandomPerspective(),
                    ])
                ),
                "group2": dict(
                    tasks=["color_saturation", "shot_framing"],
                    transforms=Tfms.Compose([
                        Tfms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                        Tfms.RandomPerspective(),
                        Tfms.RandomHorizontalFlip(),
                        Tfms.RandomVerticalFlip(),
                    ])
                )
            }
        import icevision.tfms as tfms
        detection_transforms = tfms.A.Adapter([
                tfms.A.Normalize(),
                tfms.A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
                tfms.A.PadIfNeeded(img_H, img_W, border_mode=cv2.BORDER_CONSTANT),
            ])

        dset = HybridAugmentationsRecordDataset(
            records=records,
            classification_transforms_groups=classification_transforms_groups,
            detection_transforms=detection_transforms,
        )

    Returned Record Example:
        Note that unlike the input record, each task has an `Image` attribute which
        is after the transforms have been applied. In the dataloader, these task specific
        images must be used, and the `record.common.img` is just the original image
        untransformed that shouldn't be used to train the model

        BaseRecord

        common:
            - Image ID: 4
            - Filepath: sample_image.png
            - Image: 640x640x3 <np.ndarray> Image
            - Image size ImgSize(width=640, height=640)
        color_saturation:
            - Image: 640x640x3 <np.ndarray> Image
            - Class Map: <ClassMap: {'desaturated': 0, 'neutral': 1}>
            - Labels: [1]
        shot_composition:
            - Class Map: <ClassMap: {'balanced': 0, 'center': 1}>
            - Labels: [1]
            - Image: 640x640x3 <np.ndarray> Image
        detection:
            - BBoxes: [<BBox (xmin:29, ymin:91, xmax:564, ymax:625)>]
            - Image: 640x640x3 <np.ndarray> Image
            - Class Map: <ClassMap: {'background': 0, 'person': 1}>
            - Labels: [1]
        shot_framing:
            - Class Map: <ClassMap: {'01-wide': 0, '02-medium': 1, '03-closeup': 2}>
            - Labels: [3]
            - Image: 640x640x3 <np.ndarray> Image
    """

    def __init__(
        self,
        records: List[dict],
        classification_transforms_groups: dict,
        detection_transforms: Optional[tfms.Transform] = None,
        norm_mean: Collection[float] = [0.485, 0.456, 0.406],
        norm_std: Collection[float] = [0.229, 0.224, 0.225],
        debug: bool = False,
    ):
        "Return `PIL.Image` when `debug=True`"
        self.records = records
        self.classification_transforms_groups = classification_transforms_groups
        self.detection_transforms = detection_transforms
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.debug = debug
        self.validate()

    def validate(self):
        """
        Input args validation
        * Ensure that each value in the `classification_transforms_groups` dict
          has a "tasks" and "transforms" key
        * Ensure the number of tasks mentioned in `classification_transforms_groups`
          match up _exactly_ with the tasks in the record
        """
        for group in self.classification_transforms_groups.values():
            assert set(group.keys()).issuperset(
                ["tasks", "transforms"]
            ), f"Invalid keys in `classification_transforms_groups`"

        missing_tasks = []
        record = self.load_record(0)
        for attr in flatten(
            [g["tasks"] for g in self.classification_transforms_groups.values()]
        ):
            if not hasattr(record, attr):
                missing_tasks += [attr]
        if not missing_tasks == []:
            raise ValueError(
                f"`classification_transforms_groups` has more groups than are present in the `record`. \n"
                f"Missing the following tasks: {missing_tasks}"
            )

    def __len__(self):
        return len(self.records)

    def load_record(self, i: int):
        """
        Simple record loader. Externalised for easy subclassing for custom behavior
        like loading cached records from disk
        """
        return self.records[i].load()

    def __getitem__(self, i):
        record = self.load_record(i)

        # Keep a copy of the orig img as it gets modified by albu
        original_img = deepcopy(record.img)
        if isinstance(original_img, np.ndarray):
            original_img = PIL.Image.fromarray(original_img)

        # Do detection transform and assign it to the detection task
        if self.detection_transforms is not None:
            record = self.detection_transforms(record)

        record.add_component(ImageRecordComponent(Task("detection")))
        record.detection.set_img(record.img)

        if self.debug:
            print(f"Fetching Item #{i}")

        # Do classification transforms
        for group in self.classification_transforms_groups.values():
            img_tfms = group["transforms"]
            tfmd_img = img_tfms(original_img)
            if self.debug:
                print(f"  Group: {group['tasks']}, ID: {id(tfmd_img)}")

            # NOTE:
            # Setting the same img twice (to diff parts in memory) but it's ok cuz we will unload the record later
            for task in group["tasks"]:
                # record.add_component(ImageRecordComponent(Task(task))) # TODO FIXME: This throws a weird error idk why
                comp = getattr(record, task)
                comp.add_component(ImageRecordComponent())
                comp.set_img(tfmd_img)
                if self.debug:
                    print(f"   - Task: {task}, ID: {id(tfmd_img)}")

        # This is a bit verbose, but allows us to return PIL images for easy debugging.
        # Else, it returns normalized numpy arrays, like usual icevision datasets
        for comp in record.components:
            if isinstance(comp, ImageRecordComponent):
                # Convert to `np.ndarray` if it isn't already
                if isinstance(comp.img, PIL.Image.Image):
                    comp.set_img(np.array(comp.img))
                if self.debug:  # for debugging only
                    comp.set_img(PIL.Image.fromarray(comp.img))
                else:
                    comp.set_img(
                        normalize(comp.img, mean=self.norm_mean, std=self.norm_std)
                    )

        return record

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items and {len(self.classification_transforms_groups)+1} groups>"
