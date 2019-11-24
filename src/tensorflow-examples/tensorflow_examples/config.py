import logging
import os

LOGGING_CONFIG_DICT = dict(
    version=1,
    formatters={
        "simple": {
            "format": """%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"""
        }
    },
    handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    root={"handlers": ["console"], "level": logging.DEBUG},
)

HOME_DIR = os.path.expanduser("~")

DATA_DIR = os.path.join(HOME_DIR, "tensorflow-example-data")
