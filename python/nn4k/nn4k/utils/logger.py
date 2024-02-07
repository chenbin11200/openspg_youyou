# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

import logging
from logging.config import fileConfig
from pathlib import Path

DEFAULT_LOGGER_KEY = "NN4K"
LOGGING_CONFIG_PATH = Path.joinpath(Path(__file__).parent, "logger.ini")

fileConfig(LOGGING_CONFIG_PATH)
logger = logging.getLogger(DEFAULT_LOGGER_KEY)
