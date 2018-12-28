import os
import tempfile
import shutil
import time
import logging
import signal
import importlib
import importlib.util

import numpy as np
import tensorflow as tf

class SavedModelBuilderHook(tf.train.SessionRunHook):
    """SavedModelの出力する"""
    def __init__(self, export_dir, signature_def_map, tags=None):
        self.export_dir = export_dir
        self.signature_def_map = signature_def_map
        self.tags = tags if tags is not None else [tf.saved_model.tag_constants.SERVING]

    def end(self, session):
        session.graph._unsafe_unfinalize()
        builder = tf.saved_model.builder.SavedModelBuilder(self.export_dir)
        builder.add_meta_graph_and_variables(
            session,
            self.tags,
            signature_def_map=self.signature_def_map
        )
        builder.save()