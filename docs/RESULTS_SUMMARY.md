# 心律失常预测模型 - 实验结果总结

**生成时间**: 2025-12-19  
**总实验数**: 192 条记录  
**模型数量**: 13  
**训练模式**: big, small, small_no_AT, small_refined

---

## 1. 最佳模型性能总结


### SMALL_REFINED 模式

| 排名 | 模型 | Micro-F1 | Macro-F1 | AUC |
|------|------|----------|----------|-----|
| 🥇 1 | LightGBM | 0.695 | 0.583 | 0.852 |
| 🥈 2 | XGBoost | 0.695 | 0.583 | 0.853 |
| 🥉 3 | Voting | 0.692 | 0.566 | 0.858 |
|    4 | RandomForest | 0.677 | 0.539 | 0.857 |
|    5 | MLP | 0.675 | 0.618 | 0.802 |

### BIG 模式

| 排名 | 模型 | Micro-F1 | Macro-F1 | AUC |
|------|------|----------|----------|-----|
| 🥇 1 | Stacking | 0.615 | 0.566 | 0.811 |
| 🥈 2 | XGBoost | 0.606 | 0.555 | 0.803 |
| 🥉 3 | LightGBM | 0.604 | 0.554 | 0.803 |
|    4 | Voting | 0.590 | 0.524 | 0.810 |
|    5 | RandomForest | 0.573 | 0.487 | 0.808 |

### SMALL_NO_AT 模式

| 排名 | 模型 | Micro-F1 | Macro-F1 | AUC |
|------|------|----------|----------|-----|
| 🥇 1 | LightGBM | 0.549 | 0.384 | 0.795 |
| 🥈 2 | Stacking | 0.546 | 0.377 | 0.803 |
| 🥉 3 | XGBoost | 0.544 | 0.369 | 0.795 |
|    4 | Voting | 0.534 | 0.336 | 0.804 |
|    5 | TabNet | 0.517 | 0.313 | 0.766 |

### SMALL 模式

| 排名 | 模型 | Micro-F1 | Macro-F1 | AUC |
|------|------|----------|----------|-----|
| 🥇 1 | Stacking | 0.522 | 0.282 | 0.752 |
| 🥈 2 | LightGBM | 0.522 | 0.287 | 0.741 |
| 🥉 3 | XGBoost | 0.520 | 0.287 | 0.742 |
|    4 | Voting | 0.509 | 0.253 | 0.757 |
|    5 | TabNet | 0.507 | 0.263 | 0.726 |

---

## 2. 深度学习模型性能


| 模式 | 模型 | Micro-F1 | AUC | 状态 |
|------|------|----------|-----|------|
| big | FT-Transformer | 0.000 | 0.500 | ❌ 失败 |
| big | MLP | 0.536 | 0.760 | ✅ |
| big | NODE | 0.000 | 0.500 | ❌ 失败 |
| big | TabNet | 0.532 | 0.775 | ✅ |
| small | FT-Transformer | 0.000 | 0.500 | ❌ 失败 |
| small | MLP | 0.434 | 0.721 | ✅ |
| small | NODE | 0.000 | 0.500 | ❌ 失败 |
| small | TabNet | 0.507 | 0.726 | ✅ |
| small_no_AT | FT-Transformer | 0.000 | 0.500 | ❌ 失败 |
| small_no_AT | MLP | 0.480 | 0.761 | ✅ |
| small_no_AT | NODE | 0.000 | 0.500 | ❌ 失败 |
| small_no_AT | TabNet | 0.517 | 0.766 | ✅ |
| small_refined | FT-Transformer | 0.000 | 0.500 | ❌ 失败 |
| small_refined | MLP | 0.675 | 0.802 | ✅ |
| small_refined | NODE | 0.000 | 0.500 | ❌ 失败 |
| small_refined | TabNet | 0.603 | 0.815 | ✅ |

---

## 3. SHAP特征重要性分析


**Top 10 最重要特征**:

1. **LA**
2. **INR**
3. **年龄**
4. **BNP**
5. **CHoL**
6. **UA(umol/L)**
7. **腰围**
8. **TSH**
9. **LVDd**
10. **LDL**

**临床解释**:
- **LA (左心房大小)**: 心律失常的最强预测因子
- **INR (凝血功能)**: 与房颤抗凝治疗相关
- **年龄**: 心律失常风险随年龄增加
- **BNP**: 心衰和房颤的重要标志物

---

## 4. 关键发现


### 4.1 最佳配置

- **推荐模式**: small_refined
- **推荐模型**: XGBoost 或 LightGBM
- **性能**: Micro-F1 = 0.695, AUC = 0.853
- **诊断类别**: AF/AFL, AVNRT, AVRT

### 4.2 模型失败分析

- **FT-Transformer**: 所有模式失败 (F1=0.0)
- **NODE**: 所有模式失败 (F1=0.0)
- **原因**: 概率提取逻辑错误，已在GPU1实验中修复
- **GPU1结果**: FT-Large达到F1=0.532，说明模型可用

### 4.3 小样本类别处理

- **AT类别**: F1仅0.008，建议过滤
- **PVCs类别**: F1约0.26，性能一般
- **策略**: small_refined模式过滤后性能显著提升

---

## 5. 模型推荐


### 生产环境

```python
# 推荐配置
mode = 'small_refined'
model = 'XGBoost'  # 或 'LightGBM'
expected_f1 = 0.695
expected_auc = 0.853
classes = ['AF/AFL', 'AVNRT', 'AVRT']
```

### 研究场景

- 如需更多类别，使用 `small_no_AT` 模式
- 如需解释性，考虑 TabNet (F1=0.603)
- 深度学习模型需先修复

---

## 6. 下一步建议

1. ✅ 使用XGBoost/LightGBM部署生产系统
2. 🔧 修复FT-Transformer/NODE的概率提取bug
3. 📊 在独立数据集上进行外部验证
4. 🏥 与临床医生合作调优决策阈值