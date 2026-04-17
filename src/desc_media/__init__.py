"""A Python CLI tool that uses a local LLAMA model to describe media."""

__version__ = "0.0.2"

import logging

import coloredlogs
import verboselogs

verboselogs.install()
coloredlogs.install()

ROOT_LOGGER = logging.getLogger()
