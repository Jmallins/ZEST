#!/usr/bin/env python
#Copyright 2020 Google LLC.
#SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(name='OpenNMT-py',
      description='A python implementation of OpenNMT',
      version='0.8.2',

      packages=['onmt', 'onmt.encoders', 'onmt.modules', 'onmt.tests',
                'onmt.translate', 'onmt.decoders', 'onmt.inputters',
                'onmt.models', 'onmt.utils'])
