#!/usr/bin/env python3
"""
URDF 数据加载
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_urdf(path: str) -> str:
    """从文件读取 URDF 字符串。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_spirit01_urdf() -> str:
    """获取内置的 Spirit01 机器人 URDF 字符串。"""
    return load_urdf(os.path.join(_DATA_DIR, "spirit01.urdf"))
