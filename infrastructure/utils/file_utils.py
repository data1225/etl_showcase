import pandas as pd
import numpy as np
from typing import List

def save_large_dataframe_to_excel(df: pd.DataFrame, file_path: str):
    """
    將過大的 DataFrame 分割成多個工作表，並儲存到一個 Excel 檔案中。

    Args:
        df (pd.DataFrame): 要儲存的 DataFrame。
        file_path (str): 輸出的 Excel 檔案路徑，例如 'large_data.xlsx'。
    """
    # Excel 單一工作表的最大行數
    EXCEL_ROW_LIMIT = 1048576

    # 計算需要多少個工作表來儲存整個 DataFrame
    num_sheets = np.ceil(len(df) / EXCEL_ROW_LIMIT).astype(int)
    
    # 檢查是否需要分割
    if num_sheets <= 1:
        print(f"DataFrame 大小合適，將直接寫入單一工作表。")
        df.to_excel(file_path, index=False)
        print(f"所有資料已成功寫入至檔案 '{file_path}'。")
        return
        
    print(f"DataFrame 行數 ({len(df)}) 超過 Excel 上限，將分割成 {num_sheets} 個工作表。")
    
    # 使用 ExcelWriter 來管理多個工作表的寫入
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for i in range(num_sheets):
            start_row = i * EXCEL_ROW_LIMIT
            end_row = (i + 1) * EXCEL_ROW_LIMIT
            
            # 從 DataFrame 中切分出當前工作表的資料
            chunk = df.iloc[start_row:end_row]
            
            sheet_name = f'Sheet_{i + 1}'
            print(f"正在寫入 '{sheet_name}'，包含 {len(chunk)} 行資料...")
            
            # 將切分後的資料塊寫入新的工作表
            chunk.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"所有資料已成功寫入至檔案 '{file_path}'。")


def read_and_combine_excel_sheets(file_path: str) -> pd.DataFrame:
    """
    讀取一個 Excel 檔案中的所有工作表，並將它們的內容合併成單一 DataFrame。
    
    Args:
        file_path (str): 欲讀取的 Excel 檔案路徑。
        
    Returns:
        pd.DataFrame: 包含所有工作表資料的合併後 DataFrame。
    """
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        # 建立一個列表來存放每個工作表的 DataFrame
        all_dfs: List[pd.DataFrame] = []
        
        print(f"正在讀取檔案 '{file_path}'，共包含 {len(sheet_names)} 個工作表。")
        
        # 遍歷所有工作表名稱並讀取資料
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            all_dfs.append(df)
            print(f"已讀取工作表 '{sheet_name}'，包含 {len(df)} 行資料。")
            
        # 使用 pd.concat 將所有 DataFrame 合併起來
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"所有工作表已成功合併，總行數為 {len(combined_df)}。")
        return combined_df
    
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認路徑是否正確。")
        return pd.DataFrame()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        return pd.DataFrame()