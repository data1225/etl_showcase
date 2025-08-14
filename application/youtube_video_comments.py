#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 方便修改py檔後，能即時載入py檔最新程式
# %load_ext autoreload
# %autoreload 2

from path_setup import setup_project_root
root = setup_project_root()

import re
from etl_showcase.domain.models import StatusCode
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_LOGS_SHEET_NAME,
    YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
    YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN
)
from etl_showcase.infrastructure.datasource.youtube_api import youtube_search_comments
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    create_google_sheet,
    is_sheet_exists,
    update_full_google_sheet,
    update_log_of_google_sheet,
    get_log_from_google_sheet
)

# 取得最後儲存的 next_page_token
last_saved_next_page_token = get_log_from_google_sheet(
    spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
    sheet_name=YOUTUBE_LOGS_SHEET_NAME,
    search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN
)

# variables for search comments
youtube_video_ids = ['_VB39Jo8mAQ']
screen_work_name = 'test'

# search youtube comments (自動續抓模式)
search_youtube_result = youtube_search_comments(
    video_ids=youtube_video_ids,
    current_next_page_token=last_saved_next_page_token,
    mode="auto"
)
# 測試模式呼叫（限 2 頁）：
# search_youtube_result = youtube_search_comments(video_ids=youtube_video_ids, max_page=2, mode="limit_max_page")

# update log and data in google sheets
write_secret_json()
try:
    log_content = f'Search youtube comments result: [{search_youtube_result.status_code}] {search_youtube_result.message}'
    print(log_content)
    update_log_of_google_sheet(
        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
        sheet_name=YOUTUBE_LOGS_SHEET_NAME,
        search_keyword=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
        update_content=log_content
    )

    # 如果抓取失敗且 message 中有 next_page_token，更新到 Google Sheet
    if search_youtube_result.status_code != StatusCode.SUCCESS:
        match = re.search(r'\[final next_page_token:(.*?)\]', search_youtube_result.message)
        if match:
            extracted_token = match.group(1)
            update_log_of_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN,
                update_content=extracted_token
            )

    comments = search_youtube_result.content
    if len(comments) > 0:
        # create sheet
        if not is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=screen_work_name):
            create_google_sheet(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=screen_work_name)

        # update sheet
        update_rows = [['ID', 'Parent ID', 'Level', 'Text', 'Like Count']]
        for comment in comments:
            update_rows.append([
                comment.id,
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
        if update_sheet_result.status_code == StatusCode.SUCCESS:
            # 清空 next_page_token
            update_log_of_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                search_keyword=YOUTUBE_SEARCH_COMMENTS_CURRENT_NEXT_PAGE_TOKEN,
                update_content=""
            )
        else:
            update_log_of_google_sheet(
                spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                sheet_name=YOUTUBE_LOGS_SHEET_NAME,
                search_keyword=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
                update_content=log_content
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




