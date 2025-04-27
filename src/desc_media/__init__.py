from __future__ import annotations

import logging

import coloredlogs
import verboselogs

verboselogs.install()
coloredlogs.install()

ROOT_LOGGER = logging.getLogger()
