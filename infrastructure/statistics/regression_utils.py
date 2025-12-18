import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import List

def run_multivariate_regression(
    data: pd.DataFrame, 
    x_columns: List[str],
    y_column: str,
) -> RegressionResultsWrapper:
    """
    執行多變數迴歸分析 (Ordinary Least Squares, OLS)。

    Args:
        data: 包含所有資料的 Pandas DataFrame。
        y_col: 應變數 (Y) 的欄位名稱。
        x_cols: 自變數 (X) 的欄位名稱列表。

    Returns:
        statsmodels 的迴歸結果物件。
    """
    print(f"\n--- 正在執行迴歸分析 ---")
    print(f"應變數 (Y): {y_column}")
    print(f"自變數 (X): {x_columns}")

    # 準備 X 和 Y 數據
    X = data[x_columns]
    Y = data[y_column]

    # 加入常數項 (Intercept) 到 X，這是 OLS 模型的標準步驟
    X = sm.add_constant(X)

    # 建立並擬合 OLS 模型
    model = sm.OLS(Y, X)
    results = model.fit()

    return results

def display_regression_coefficients(results: RegressionResultsWrapper):
    """取得迴歸係數。"""
    # 係數表格：直接從 results 物件獲取 params (係數) 和 pvalues (P值)
    coef_df = pd.DataFrame({
        'coef': results.params,
        'pvalues': results.pvalues
    })
    
    # 為了清晰顯示，將 DataFrame 轉換為 Markdown 格式
    print(coef_df.to_markdown())

    adj_r2_val = getattr(results, 'rsquared_adj', 0.0)
    print(f"調整後 R 平方: {adj_r2_val:.4f}")

    print("\n解讀：coef 表示該變數每增加一個單位，Y (房屋價值中位數) 預期增加或減少的數量。")
    print("pvalues 用於判斷係數的統計顯著性，P值越小越顯著。")
    print("R平方 用於衡量模型中的自變數（X）能解釋應變數（Y）變異的百分比，如R平方 = 0.75，代表模型針對Y這個現象具75%的解釋力。")

def get_regression_coefficients(results: RegressionResultsWrapper)-> pd.DataFrame:
    """
    取得迴歸係數相關重要資訊。
    """
    # 1. 建立基礎數據字典
    data_dict = {
        'coef': results.params,
        'pvalues': results.pvalues,
    }
    
    # 建立基礎 DataFrame
    coef_df = pd.DataFrame(data_dict)

    # 2. 建立 MultiIndex (多重索引) 以實現 Excel 中的合併欄位效果
    # 確保 adj_r2_label 在 multi_cols 定義前已生成
    adj_r2_val = getattr(results, 'rsquared_adj', 0.0)
    adj_r2_label = f"調整後 R 平方: {adj_r2_val:.4f}"
    
    # 正確定義 multi_cols
    multi_cols = pd.MultiIndex.from_tuples([
        (adj_r2_label, 'coef'),
        (adj_r2_label, 'pvalues')
    ])
    
    # 套用多重索引到 DataFrame 欄位
    coef_df.columns = multi_cols

    return coef_df

def generate_regression_predictions(
    df_original: pd.DataFrame, 
    results: RegressionResultsWrapper,
    index_column: str,
    x_columns: List[str],
    y_column: str,
) -> pd.DataFrame:
    """創建包含實際值和預測值的 dataframe。"""

    # 計算預測值
    X_for_pred = sm.add_constant(df_original[x_columns])
    predictions = results.predict(X_for_pred)

    # 建立包含實際值和預測值的新 DataFrame
    results_df = pd.DataFrame({
        'index': df_original[index_column], 
        'true values': df_original[y_column],
        'predicted values': predictions
    })

    return results_df