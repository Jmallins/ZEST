"""Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Modules for translation """
from onmt.translate.translator import Translator
from onmt.translate.translation import Translation, TranslationBuilder
from onmt.translate.beam import Beam, GNMTGlobalScorer
from onmt.translate.beam_search import BeamSearch
from onmt.translate.decode_strategy import DecodeStrategy
from onmt.translate.random_sampling import RandomSampling
from onmt.translate.penalties import PenaltyBuilder
from onmt.translate.translation_server import TranslationServer, \
    ServerModelError

__all__ = ['Translator', 'Translation', 'Beam', 'BeamSearch',
           'GNMTGlobalScorer', 'TranslationBuilder',
           'PenaltyBuilder', 'TranslationServer', 'ServerModelError',
           "DecodeStrategy", "RandomSampling"]
