__all__ = ["UnetCallback"]

from icevision.imports import *


class UnetCallback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1

        self.learn.records = self.yb[0]
        self.learn.yb = (self.xb[0][1],)
        self.learn.xb = (self.xb[0][0],)
