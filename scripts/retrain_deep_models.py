#!/usr/bin/env python3
"""
重新训练FT-Transformer和NODE模型
使用优化后的参数配置
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# 导入 run_analysis.py 的所有函数和类
sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import (
    prepare_data, get_model, plot_dca, plot_calibration_curve_func,
    OneVsRestClassifier, f1_score, recall_score, precision_score, roc_auc_score,
    np, xgb, shap, plt, re
)

if __name__ == "__main__":
    # 配置
    DEEP_MODELS = ['FT-Transformer', 'NODE']  # 只训练失败的模型
    TEST_MODE = 'big'  # 先在big模式测试
    
    print("="*70)
    print("🔧 重新训练深度学习模型 (优化配置)")
    print("="*70)
    print(f"模型: {DEEP_MODELS}")
    print(f"测试模式: {TEST_MODE}")
    print(f"优化项:")
    print(f"  - Epochs: 20 → 100")
    print(f"  - Batch Size: 128 → 256")
    print(f"  - Learning Rate: 1e-3 → 1e-4/5e-4")
    print(f"  - Early Stopping: Yes (patience=20)")
    print(f"  - Checkpointing: Yes")
    print("="*70)
    
    # 加载数据
    print(f"\n📊 加载数据模式: {TEST_MODE}")
    X_train, Y_train, X_val, Y_val, mlb, feature_names = prepare_data(TEST_MODE)
    print(f"✓ 训练集: {X_train.shape[0]} 样本")
    print(f"✓ 验证集: {X_val.shape[0]} 样本")
    print(f"✓ 特征数: {len(feature_names)}")
    print(f"✓ 类别: {list(mlb.classes_)}")
    
    # 准备保存结果
    results = []
    results_csv_path = f"outputs/retrain_results_{TEST_MODE}.csv"
    
    # 训练循环
    for model_idx, model_name in enumerate(DEEP_MODELS):
        print(f"\n{'='*70}")
        print(f"🎯 [{model_idx+1}/{len(DEEP_MODELS)}] 训练 {model_name}")
        print(f"{'='*70}")
        print(f"⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        try:
            base_clf = get_model(model_name)
            if base_clf is None:
                print(f"❌ {model_name} 不可用")
                continue
            
            print(f"📦 模型初始化: {type(base_clf).__name__}")
            print(f"🔄 开始 OneVsRest 训练 ({len(mlb.classes_)} 个二分类器)...")
            print(f"   (预计需要 5-10 分钟,请耐心等待...)")
            
            clf = OneVsRestClassifier(base_clf, verbose=1)
            clf.fit(X_train, Y_train)
            
            elapsed = time.time() - start_time
            print(f"\n✅ 训练完成! 用时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
            
            # 预测
            print(f"🔮 开始预测...")
            Y_prob = clf.predict_proba(X_val)
            Y_pred = (Y_prob >= 0.5).astype(int)
            print(f"✓ 预测完成")
            
            # 整体指标
            print(f"\n📊 计算整体指标...")
            micro_f1 = f1_score(Y_val, Y_pred, average='micro')
            macro_f1 = f1_score(Y_val, Y_pred, average='macro')
            try:
                micro_auc = roc_auc_score(Y_val, Y_prob, average='micro')
            except:
                micro_auc = None
            
            print(f"  ⭐ Micro-F1: {micro_f1:.4f}")
            print(f"  ⭐ Macro-F1: {macro_f1:.4f}")
            if micro_auc:
                print(f"  ⭐ Micro-AUC: {micro_auc:.4f}")
            
            # 逐类指标
            print(f"\n📊 逐类别指标:")
            print(f"{'Class':<25} {'AUC':<8} {'F1':<8} {'Recall':<8} {'Precision':<8}")
            print("-"*70)
            
            for i, class_name in enumerate(mlb.classes_):
                y_true_i = Y_val[:, i]
                y_prob_i = Y_prob[:, i]
                y_pred_i = Y_pred[:, i]
                
                # AUC
                try:
                    auc = roc_auc_score(y_true_i, y_prob_i) if len(np.unique(y_true_i)) > 1 else None
                except:
                    auc = None
                
                f1 = f1_score(y_true_i, y_pred_i)
                rec = recall_score(y_true_i, y_pred_i)
                prec = precision_score(y_true_i, y_pred_i, zero_division=0)
                
                auc_str = f"{auc:.3f}" if auc is not None else "N/A"
                print(f"{class_name:<25} {auc_str:<8} {f1:<8.3f} {rec:<8.3f} {prec:<8.3f}")
                
                results.append({
                    'Label_Mode': TEST_MODE,
                    'Model': model_name,
                    'Class': class_name,
                    'AUC': auc,
                    'F1': f1,
                    'Recall': rec,
                    'Precision': prec,
                    'Micro_F1': micro_f1,
                    'Macro_F1': macro_f1,
                    'Training_Time_Sec': elapsed
                })
            
            # 保存结果
            pd.DataFrame(results).to_csv(results_csv_path, index=False)
            print(f"\n💾 结果已保存: {results_csv_path}")
            
            # 成功判定
            if micro_f1 > 0.1 and (micro_auc is None or micro_auc > 0.6):
                print(f"\n✅ ✅ ✅ {model_name} 训练成功! F1={micro_f1:.3f} ✅ ✅ ✅")
            else:
                print(f"\n⚠️ {model_name} 训练完成但性能仍然较低 (F1={micro_f1:.3f})")
            
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终总结
    print(f"\n{'='*70}")
    print(f"🎉 重训练完成!")
    print(f"{'='*70}")
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📈 性能摘要:")
        summary = results_df.groupby('Model').agg({
            'F1': 'mean',
            'AUC': 'mean',
            'Training_Time_Sec': 'sum'
        }).round(3)
        print(summary)
        print(f"\n💾 详细结果: {results_csv_path}")
    print(f"{'='*70}")
