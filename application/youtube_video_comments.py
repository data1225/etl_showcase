#!/usr/bin/env python
# coding: utf-8

# In[13]:


from path_setup import setup_project_root
root = setup_project_root()

from film_tv_animation_insights.domain.models import StatusCode
from film_tv_animation_insights.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_LOGS_SHEET_NAME,
    YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
)
from film_tv_animation_insights.infrastructure.youtube_api import youtube_search_comments
from film_tv_animation_insights.infrastructure.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    create_google_sheet,
    is_sheet_exists,
    update_full_google_sheet,
    update_log_of_google_sheet,
)


# variables for search comments
youtube_video_ids = ['_VB39Jo8mAQ']
screen_work_name = 'test'

# search youtube comments
search_youtube_result = youtube_search_comments(
    video_ids = youtube_video_ids, 
    max_comment_count_per_page = 100, 
    max_page = 1
)

# update log and data in google sheets
write_secret_json()
try:
    log_content = f'Search youtube comments result: [{search_youtube_result.status_code}] {search_youtube_result.message}'
    print(log_content)
    update_log_of_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = YOUTUBE_LOGS_SHEET_NAME,
        search_keyword = YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
        update_content = log_content
    )

    comments = search_youtube_result.content
    if len(comments) > 0:
        # create sheet
        if is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=screen_work_name) == False:   
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
            spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
            sheet_name = screen_work_name,
            update_rows = update_rows
        )
        log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
        print(log_content)
        if update_sheet_result.status_code != StatusCode.SUCCESS:
            update_log_of_google_sheet(
                spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
                sheet_name = YOUTUBE_LOGS_SHEET_NAME,
                search_keyword = YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
                update_content = log_content
            )  
    else:
        log_content = f'There is no comment for screen work:{screen_work_name}'
        print(log_content)
        update_log_of_google_sheet(
            spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
            sheet_name = YOUTUBE_LOGS_SHEET_NAME,
            search_keyword = YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
            update_content = log_content
        )        
finally:
    delete_secret_json()


# In[ ]:




