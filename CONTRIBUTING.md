# Contributing

请不要提交 DexterHand 数据、MANO pickle、任何派生 NPZ 或 Rerun recording。新增功能需要维持以下边界：只支持右手 MANO、只输出五个末端指节、只针对解析 Cuboid。

提交前运行：

```bash
python -m pytest
python -m compileall -q src tests
git diff --check
```

Issues 请包含 Python、Torch、smplx、DexterCap commit、输入 metadata（去除私有路径）及可复现命令。
