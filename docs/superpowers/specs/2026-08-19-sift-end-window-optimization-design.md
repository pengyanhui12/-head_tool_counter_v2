# SIFT缓存与尾窗预筛选优化设计

## 目标

在不改变SIFT、BFMatcher、RANSAC及单应矩阵验收阈值的前提下，减少重复特征提取，并把尾窗30帧的全量SIFT评分缩减为6帧。优化必须保持主流程检测和关联规则不变，并可通过配置回退到旧的尾窗全量匹配行为。

## SIFT特征缓存

`FeatureMatcher` 内部新增小容量LRU缓存，键为输入 `numpy.ndarray` 的对象身份，同时保存图像强引用、关键点和描述子。命中时必须同时满足缓存键一致且缓存图像对象 `is` 当前图像，防止对象ID复用导致错误特征。

缓存默认保留4张最近图像，内存有明确上限；这足以覆盖“当前帧、上一关键帧和Recovery锚点”等高频复用。每次匹配按“当前帧后、目标关键帧最后”的顺序刷新LRU，使重复使用的目标关键帧不易被逐帧候选淘汰。缓存包括无关键点结果，避免反复处理低纹理画面。

流水线把 `Frame.image` 视为质量评估后的只读数据。若未来存在原地修改图像的流程，修改方必须先清除缓存或传入新数组。提供 `clear_feature_cache()` 用于显式失效。缓存只优化计算，不改变特征结果和 `MatchResult` 内容。

新增Matcher配置：

```yaml
feature_cache_size: 4
```

设为0可完全关闭缓存，作为论文消融实验和问题回退开关。

## 尾窗六帧预筛选

新增Pipeline配置：

```yaml
end_window_match_candidates: 6
```

`select_end_keyframes()` 先按 `frame_id` 排序，把尾窗均匀划分为最多6个连续时间分段；每段选择质量最高的一帧。质量排序键固定为：

```text
sharpness_score × exposure_score
→ sharpness_score
→ exposure_score
→ frame_id
```

最后一项使完全同分时优先较晚帧，保证算法确定性并倾向视频末端覆盖。随后只对这最多6帧执行原有SIFT匹配，并继续使用：

```text
sharpness_score × valid_match.num_inliers
```

选择最终最佳2帧。若候选数配置为0或大于等于尾窗帧数，则退化为旧的全量SIFT评分；空尾窗行为保持不变。

按当前196帧样本，预期调用数从：

```text
12次主流程 + 30次尾窗评分 + 1次尾窗复核 = 43次
```

降低为约：

```text
12次主流程 + 6次尾窗评分 + 1次尾窗复核 = 19次
```

## 配置接入

`apps/offline_scan.py` 将 `feature_cache_size` 传给 `FeatureMatcher`，将 `end_window_match_candidates` 传给 `KeyframeSelector`。默认值分别为4和6。现有本地视频路径、输出目录和其他算法配置不得修改。

## 测试与验收

- 验证同一图像重复匹配只提取一次特征，容量超限按LRU淘汰，清空后重新提取，容量0禁用缓存。
- 验证30帧只选择6个预筛候选执行SIFT，并保持时间分段覆盖和确定性。
- 验证候选配置大于等于窗口长度时仍执行旧的全量匹配。
- 完整pytest必须通过。
- 使用同一测试视频比较优化前后：正式计数、分类计数、对象ID数量和报告状态必须一致；记录Feature Matching calls、总耗时和尾窗新增关键帧。

## 非目标

本次不修改Global Mosaic，不更换ORB/FLANN，不缩放SIFT输入，不改变匹配或单应矩阵阈值，也不缓存跨进程或跨视频特征。
