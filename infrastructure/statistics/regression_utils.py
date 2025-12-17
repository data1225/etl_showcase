import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import List

def run_multivariate_regression(
    df: pd.DataFrame, 
    x_columns: List[str],
    y_column: str,
) -> RegressionResultsWrapper:
    """
    執行多變數迴歸分析 (Ordinary Least Squares, OLS)。

    Args:
        df: 包含所有資料的 Pandas DataFrame。
        y_col: 應變數 (Y) 的欄位名稱。
        x_cols: 自變數 (X) 的欄位名稱列表。

    Returns:
        statsmodels 的迴歸結果物件。
    """
    print(f"\n--- 正在執行迴歸分析 ---")
    print(f"應變數 (Y): {y_column}")
    print(f"自變數 (X): {x_columns}")

    # 準備 X 和 Y 數據
    X = df[x_columns]
    Y = df[y_column]

    # 加入常數項 (Intercept) 到 X，這是 OLS 模型的標準步驟
    X = sm.add_constant(X)

    # 建立並擬合 OLS 模型
    model = sm.OLS(Y, X)
    results = model.fit()

    return results

def display_regression_coefficients(results: RegressionResultsWrapper):
    """印出迴歸係數。"""
    # 係數表格：直接從 results 物件獲取 params (係數) 和 pvalues (P值)
    coef_df = pd.DataFrame({
        'coef': results.params,
        'P>|t|': results.pvalues
    })
    
    # 為了清晰顯示，將 DataFrame 轉換為 Markdown 格式
    print(coef_df.to_markdown())

    print("\n解讀：coef 表示該變數每增加一個單位，Y (房屋價值中位數) 預期增加或減少的數量。")
    print("P>|t| (P值) 用於判斷係數的統計顯著性，P值越小越顯著。")

def generate_regression_predictions(
    df_original: pd.DataFrame, 
    results: RegressionResultsWrapper,
    x_columns: List[str],
    y_column: str,
) -> pd.DataFrame:
    """創建包含實際值和預測值的 dataframe。"""

    # 計算預測值
    X_for_pred = sm.add_constant(df_original[x_columns])
    predictions = results.predict(X_for_pred)

    # 建立包含實際值和預測值的新 DataFrame
    results_df = pd.DataFrame({
        'true values': df_original[y_column],
        'predicted values': predictions
    })

    return results_df