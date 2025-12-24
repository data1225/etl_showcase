import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 解決 Matplotlib 中文顯示問題 (針對常見字體設定)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 


def plot_residual_analysis(df, true_col, pred_col, title="殘差分析圖"):
    """
    輸入包含實際值與預測值的 DataFrame，繪製殘差圖。
    
    參數:
    df: pd.DataFrame
    true_col: 實際值欄位名稱
    pred_col: 預測值欄位名稱
    """
    
    # 1. 計算殘差 (Residuals = Actual - Predicted)
    # 有些定義會使用 Predicted - Actual，但在統計診斷中通常使用 Actual - Predicted
    df = df.copy()
    df['residual'] = df[true_col] - df[pred_col]
    
    # 2. 計算基本統計量
    res_mean = df['residual'].mean()
    res_std = df['residual'].std()
    
    # 3. 繪圖設定
    plt.figure(figsize=(10, 6))
    
    # 使用 Seaborn 繪製散點圖
    sns.scatterplot(x=df[pred_col], y=df['residual'], alpha=0.6, edgecolors='w')
    
    # 4. 加入 Y=0 的橫線 (理想情況殘差應隨機分佈在 0 附近)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='基準線 (Y=0)')
    
    # 5. 加入平均殘差線 (觀察是否有整體偏誤)
    plt.axhline(y=res_mean, color='blue', linestyle=':', alpha=0.5, label=f'平均殘差: {res_mean:.2f}')
    
    # 6. 設定標題與標籤
    plt.title(title, fontsize=16)
    plt.xlabel('預測值 (Predicted Values)', fontsize=12)
    plt.ylabel('殘差 (Residuals: Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # 7. 顯示圖表
    plt.tight_layout()
    plt.show()