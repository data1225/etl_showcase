#!/usr/bin/env python
# coding: utf-8

# In[3]:


from path_setup import setup_project_root
root = setup_project_root()

from film_tv_animation_insights.domain.models import BaseResponse, StatusCode
from film_tv_animation_insights.domain.youtube_models import YoutubeVideo, Topic
from film_tv_animation_insights.infrastructure.time_utils import get_previous_month_range_in_utc
from film_tv_animation_insights.infrastructure.youtube_api import youtube_search_videos
from film_tv_animation_insights.infrastructure.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    update_full_google_sheet,
    update_log_of_google_sheet,
)
from film_tv_animation_insights.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME,
    YOUTUBE_LOGS_SHEET_NAME,
)


# search youtube videos
search_youtube_result = BaseResponse[YoutubeVideo](
    status_code=StatusCode.WAIT_FOR_PROCESS,
    message='',
    content=None
)    
start_utc_datetime, end_utc_datetime = get_previous_month_range_in_utc()
topics = [
    Topic('心理議題', ['心理學', '自我成長', '心情', '感情']),
    Topic('社會弱勢議題', ['社會弱勢', '經濟弱勢', '社會不平等']),
    Topic('社會議題', ['社會議題', '社會問題']),
    Topic('科技議題', ['AI', '科技']),
]
for topic in topics:
    for keyword in topic.keywords:
        search_youtube_result = youtube_search_videos(keyword, start_utc_datetime, end_utc_datetime)
        if search_youtube_result.content is None:
            break
        topic.add_youtube_videos(search_youtube_result.content)

# update log and data in google sheets
write_secret_json()
try:
    log_content = f'Search youtube videos result: [{search_youtube_result.status_code}] {search_youtube_result.message}'
    print(log_content)
    update_log_of_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = YOUTUBE_LOGS_SHEET_NAME,
        search_keyword = YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME,
        update_content = log_content
    )
    
    # transform original data to table
    update_rows = [["主題", "搜尋關鍵字", "影片ID", "影片標題", "影片描述"]]
    for topic in topics:
        keywords_string = '、'.join(topic.keywords)
        for video in topic.youtube_videos:
            update_rows.append([
                topic.name,
                keywords_string,
                video.id,
                video.title,
                video.description
            ])
    # update google sheet
    update_sheet_result = update_full_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME,
        update_rows = update_rows
    )
    log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
    print(log_content)
    if update_sheet_result.status_code != StatusCode.SUCCESS:
        update_log_of_google_sheet(
            spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
            sheet_name = YOUTUBE_LOGS_SHEET_NAME,
            search_keyword = YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME,
            update_content = log_content
        )        
finally:
    delete_secret_json()


# In[ ]:




