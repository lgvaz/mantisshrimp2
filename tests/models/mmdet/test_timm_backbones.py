import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "ds, model_type",
    [
        (
            "fridge_ds",
            models.mmdet.retinanet,
        ),
    ],
)
class TestTimmBackbones:
    def dls_model(self, ds, model_type, samples_source, request):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)

        # backbone = model_type.backbones.mmdet.resnet50_fpn_1x()
        backbone = model_type.backbones.timm.mobilenet.mobilenetv3_large_100
        backbone.config_path = samples_source / backbone.config_path

        model = model_type.model(backbone=backbone(pretrained=True), num_classes=5)

        return train_dl, valid_dl, model

    def test_mmdet_bbox_models_fastai(self, ds, model_type, samples_source, request):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, samples_source, request
        )

        learn = model_type.fastai.learner(
            dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params
        )
        learn.fine_tune(1, 3e-4)

    def test_mmdet_bbox_models_light(self, ds, model_type, samples_source, request):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            weights_summary=None,
            num_sanity_val_steps=0,
            logger=False,
            checkpoint_callback=False,
        )
        trainer.fit(light_model, train_dl, valid_dl)
