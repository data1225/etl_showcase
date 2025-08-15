#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 即時載入最新修改
# %load_ext autoreload
# %autoreload 2

from path_setup import setup_project_root
root = setup_project_root()

import re, os
from dotenv import load_dotenv
from etl_showcase.domain.models import StatusCode
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_LOGS_SHEET_NAME,
    YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
    YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN,
    YOUTUBE_SEARCH_COMPLETED_STATUS 
)
from etl_showcase.infrastructure.datasource.youtube_api import youtube_search_comments
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    create_google_sheet,
    is_sheet_exists,
    update_full_google_sheet,
    update_log_of_google_sheet,
    get_log_from_google_sheet,
    get_full_google_sheet 
)

load_dotenv()
write_secret_json()
try:
    # 取得最後儲存的 next_page_token
    last_saved_next_page_token = get_log_from_google_sheet(
        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
        sheet_name=YOUTUBE_LOGS_SHEET_NAME,
        search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN
    )
    
    # 如果已完成所有留言搜尋則停止
    if last_saved_next_page_token == YOUTUBE_SEARCH_COMPLETED_STATUS:
        print("已記錄指定影片的所有留言。")
        raise SystemExit
    
    # variables for search comments
    youtube_video_ids = os.getenv("YOUTUBE_SEARCH_COMMENTS_SOURCE_VIDEO_IDS", "").split(",")
    screen_work_name = os.getenv("YOUTUBE_SEARCH_COMMENTS_CLUSTER_TITLE", "")
    
    # search youtube comments (自動續抓模式)
    search_youtube_result = youtube_search_comments(
        video_ids=youtube_video_ids,
        current_next_page_token=last_saved_next_page_token,
        mode="auto"
    )
    # 測試模式呼叫（限 2 頁）
    # search_youtube_result = youtube_search_comments(video_ids=youtube_video_ids, max_page=2, mode="limit_max_page")

    # update log and data in google sheets
    log_content = f'Search youtube comments result: [{search_youtube_result.status_code}] {search_youtube_result.message}'
    print(log_content)
    update_log_of_google_sheet(
        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
        sheet_name=YOUTUBE_LOGS_SHEET_NAME,
        search_keyword=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
        update_content=log_content
    )

    comments = search_youtube_result.content
    if len(comments) > 0:
        if is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=screen_work_name):
            update_rows = get_full_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=screen_work_name
            )
            old_last_index = int(update_rows[-1][0])
        else:
            update_rows = [['ID', 'Parent ID', 'Level', 'Text', 'Like Count']]
            old_last_index = 0
            create_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=screen_work_name
            )

        # update sheet
        for comment in comments:
            update_rows.append([
                comment.id + old_last_index,
                comment.parent_id,
                comment.level,
                comment.textDisplay,
                comment.likeCount,
            ])
        update_sheet_result = update_full_google_sheet(
            spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
            sheet_name=screen_work_name,
            update_rows=update_rows
        )
        log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
        print(log_content)
        if update_sheet_result.status_code != StatusCode.SUCCESS:
            update_log_of_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                search_keyword=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
                update_content=log_content
            )

        # 確認並更新 next_page_token
        if search_youtube_result.status_code == StatusCode.SUCCESS and update_sheet_result.status_code == StatusCode.SUCCESS:
            update_log_of_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN,
                update_content=YOUTUBE_SEARCH_COMPLETED_STATUS
            )
        elif search_youtube_result.status_code != StatusCode.SUCCESS:
            match = re.search(r'\[final next_page_token:(.*?)\]', search_youtube_result.message)
            if match:
                extracted_token = match.group(1)
                update_log_of_google_sheet(
                    spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                    sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                    search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN,
                    update_content=extracted_token
                ) 

    else:
        log_content = f'There is no comment for screen work:{screen_work_name}'
        print(log_content)
        update_log_of_google_sheet(
            spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
            sheet_name=YOUTUBE_LOGS_SHEET_NAME,
            search_keyword=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
            update_content=log_content
        )
finally:
    delete_secret_json()


# In[ ]:




