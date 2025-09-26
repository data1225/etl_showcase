import os

def save_plotly_html(fig, output_path: str):
    # 確保目標資料夾存在
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"檔案已成功儲存至: {output_path}")
    
    return output_path

def save_html(html_content, output_path: str):
    """
    將 HTML 內容寫入指定的檔案路徑。
    
    Args:
        output_path (str): 檔案的完整路徑。
        html_content (str): 要寫入檔案的 HTML 內容。
    """
    # 確保目標資料夾存在
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 將內容寫入檔案
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"檔案已成功儲存至: {output_path}")
    except IOError as e:
        print(f"寫入檔案時發生錯誤: {e}")