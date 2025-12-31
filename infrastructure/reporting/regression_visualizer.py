import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# 解決 Matplotlib 中文顯示問題 (針對常見字體設定)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 


def plot_residual_analysis(df, actual_col, pred_col, title="殘差分析圖"):
    """
    輸入包含實際值與預測值的 DataFrame，繪製殘差圖。
    
    參數:
    df: pd.DataFrame
    actual_col: 實際值欄位名稱
    pred_col: 預測值欄位名稱
    """
    
    # 1. 計算殘差 (Residuals = Actual - Predicted)
    # 有些定義會使用 Predicted - Actual，但在統計診斷中通常使用 Actual - Predicted
    df = df.copy()
    df['residual'] = df[actual_col] - df[pred_col]
    
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

def plot_actual_vs_predicted(df, actual_col, pred_col, title="實際值 vs. 預測值散點圖"):
    """
    繪製實際值與預測值的散點圖，並加入 X=Y 參考線。
    
    參數:
    df: 包含數據的 pd.DataFrame
    actual_col: 實際值欄位名稱
    pred_col: 預測值欄位名稱
    """
    # 1. 提取數據並移除空值
    temp_df = df[[actual_col, pred_col]].dropna()
    y_actual = temp_df[actual_col]
    y_pred = temp_df[pred_col]

    # 2. 計算評估指標
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    # 3. 繪圖設定
    plt.figure(figsize=(10, 8))
    
    # 繪製散點
    sns.scatterplot(x=y_pred, y=y_actual, alpha=0.5, color='royalblue', edgecolor='w', label='觀測資料點')
    
    # 4. 繪製 X=Y 參考線 (理想預測線)
    max_val = max(y_actual.max(), y_pred.max())
    min_val = min(y_actual.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='完美預測線 (X=Y)')

    # 5. 在圖表上顯示統計資訊
    stats_text = f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.2f}$"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 6. 圖表裝飾
    plt.title(title, fontsize=16)    
    plt.xlabel(f'預測值: {pred_col}', fontsize=12)
    plt.ylabel(f'實際值: {actual_col}', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # 保持座標軸比例一致（選用，視數據範圍而定）
    # plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()