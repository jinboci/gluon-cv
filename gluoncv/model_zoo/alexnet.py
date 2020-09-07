# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument
"""Alexnet, implemented in Gluon."""
__all__ = ['AlexNet', 'alexnet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

# Net
class AlexNet(HybridBlock):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = x.as_np_ndarray()
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def alexnet(pretrained=False, ctx=cpu(),
            root='~/.mxnet/models', **kwargs):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = AlexNet(**kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('alexnet', tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net
