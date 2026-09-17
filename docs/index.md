---
layout: default
title: DexterHand Contact Extractor
---

# DexterHand Contact Extractor

**一个独立的 DexterHand 演示裁剪与五指末端接触提取工具。**

它把时间区间 `[start, end)` 裁剪成一段新的演示轨迹，并在其中寻找 thumb、index、middle、ring、pinky 的第三指节与轴对齐长方体的接触。手腕、掌心、MCP、PIP 永远不会输出为接触。

## Quick start

```bash
export DEXTERCAP_ROOT=~/projects/dextercap
export MANO_MODEL_PATH=$DEXTERCAP_ROOT/HandReconstruction/Data/HumanModels/mano
pip install -e .

dexterhand-contact-extractor extract --input demo-right.npz --output demo-contact.npz \
  --start 2 --end 6.5 --mano-model-path "$MANO_MODEL_PATH" --device cpu
dexterhand-contact-extractor visualize --input demo-contact.npz --dextercap-root "$DEXTERCAP_ROOT"
```

完整安装、格式契约和可视化说明位于仓库 [README](../README.md)。
