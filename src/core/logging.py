import logging
import sys

def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, datefmt=datefmt, handlers=[logging.StreamHandler(sys.stdout)])
    for lib in ("insightface", "onnxruntime", "PIL", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)

setup_logging()