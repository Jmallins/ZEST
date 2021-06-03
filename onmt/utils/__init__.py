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

"""Module defining various utilities."""
from onmt.utils.misc import split_corpus, aeq, use_gpu, set_random_seed
from onmt.utils.report_manager import ReportMgr, build_report_manager
from onmt.utils.statistics import Statistics
from onmt.utils.optimizers import MultipleOptimizer, \
    Optimizer, AdaFactor

__all__ = ["split_corpus", "aeq", "use_gpu", "set_random_seed", "ReportMgr",
           "build_report_manager", "Statistics",
           "MultipleOptimizer", "Optimizer", "AdaFactor"]
