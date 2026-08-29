import inspect
import logging
from pathlib import Path
import os

cur = Path(__file__).parent.parent
if not os.path.exists(cur / "logs"):
    os.makedirs(cur / "logs")

filted = False

if filted:
    FILTER_FILE = ['actions.py', 'search_rader.py', 'track_rader.py', 'abstract_entry.py',
                   'track_rader_state.py', 'uav.py', 'game.py', 'weapon.py', 'map.py', 'math_tool.py']
else:
    FILTER_FILE = []


def filter_log_by_file(func):
    def wrapper(self, msg, *args, **kwargs):
        # 获取真实调用文件（跳过调试器 + logger自身）
        frame = inspect.currentframe()
        while frame:
            fname = os.path.basename(frame.f_code.co_filename)
            # 跳过调试器 & 日志文件本身
            if "pydevd" not in fname and fname != os.path.basename(__file__):
                break
            frame = frame.f_back

        # 如果在过滤列表里，直接返回，不打日志
        if frame and os.path.basename(frame.f_code.co_filename) in FILTER_FILE:
            return

        # 否则执行原来的日志逻辑
        return func(self, msg, *args, **kwargs)

    return wrapper


def singleton(cls):
    _instance = {}

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return wrapper


@singleton
class AppLogger:
    def __init__(self, name="app_logger", log_file="app.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(cur / "logs" / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    # -------------- 安全增强版，绝对不传递 mute 给原生 logger --------------
    @filter_log_by_file
    def info(self, msg, *args, **kwargs):
        if 'is_in_file' in kwargs:
            kwargs.pop('is_in_file')
        # 自动提取 mute，不传给底层
        mute = kwargs.pop("mute", False)
        for ag in args:
            msg += ' ' + str(ag)
        args = []
        if not mute:
            self.logger.info(msg, *args, stacklevel=3, **kwargs)

    @filter_log_by_file
    def debug(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.debug(msg, *args, stacklevel=3, **kwargs)

    @filter_log_by_file
    def warning(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.warning(msg, *args, stacklevel=3, **kwargs)

    @filter_log_by_file
    def error(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.error(msg, *args, stacklevel=3, **kwargs)

    @filter_log_by_file
    def critical(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.critical(msg, *args, stacklevel=3, **kwargs)

    # --------------------------------------------------------------------
    def get_logger(self):
        return self


def _green_log_str(msg):
    return f'\033[32m{msg}\033[0m'
