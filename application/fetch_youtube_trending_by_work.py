#!/usr/bin/env python
# coding: utf-8

# In[5]:



#################################################################
# 此次務任為：搜集熱門影評及熱門留言，以進行男頻高流量權謀爽劇演進研究
#################################################################
    
from path_setup import setup_project_root
root = setup_project_root()

from etl_showcase.domain.models import (
    BaseResponse, 
    StatusCode,
    LanguageCode,
)
from etl_showcase.domain.youtube_models import (
    YoutubeVideo, 
    Topic,
    VideoSearchOrder,
)
from etl_showcase.infrastructure.utils.time_utils import get_previous_month_range_in_utc
from etl_showcase.infrastructure.datasource.youtube_api import youtube_search_videos
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    update_full_google_sheet,
    update_youtube_log_of_google_sheet,
    create_google_sheet,
    is_sheet_exists,
)
from etl_showcase.config.youtube import (
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
# # 縮小搜尋量供測試用，因youtube API有搜尋上限
# topics = [
#     Topic('科技議題', ['AI']),
# ]
topics = [
    Topic('藏海傳-zh', ['藏海傳 影評']),
    Topic('藏海傳-en', ['Zang Hai Zhuan review']),
    Topic('琅琊榜-zh', ['琅琊榜 影評']),
    Topic('琅琊榜-en', ['Nirvana in Fire review']),
    Topic('慶餘年-zh', ['慶餘年 影評']),
    Topic('慶餘年-en', ['Joy of Life review']),
]

for topic in topics:
    for keyword in topic.keywords:
        # Determine the language based on topic name
        if 'en' in topic.name:
            relevanceLanguage = LanguageCode.English
        else:
            relevanceLanguage = LanguageCode.Chinese

        search_youtube_result = youtube_search_videos(
            query=keyword,
            search_count=50,
            order=VideoSearchOrder.RELEVANCE,
            relevanceLanguage=relevanceLanguage
        )
        print(f'keyword: {keyword}, result: [{search_youtube_result.status_code}] {search_youtube_result.message}')
        if search_youtube_result.content is None:
            break
        topic.add_youtube_videos(search_youtube_result.content)

# update log and data in google sheets
write_secret_json()
try:
    sheet_name_1 = "男頻高流量權謀爽劇影評影片資料"
    sheet_name_2 = "男頻高流量權謀爽劇影評影片ID(方便抓留言)"
    if is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=sheet_name_1) == False:
        create_google_sheet(
            spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
            sheet_name=sheet_name_1
        )
    if is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=sheet_name_2) == False:
        create_google_sheet(
            spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
            sheet_name=sheet_name_2
        )
        
    ####### 影評影片資料
    # transform original data to table
    update_rows = [["Topic", "Search keyword", 
                    "Video ID", "Video URL", "Video title", "Video description", 
                    "Publish datetime", "Channel name", "Channel ID", "Thumbnail URL"]]
    for topic in topics:
        keywords_string = '、'.join(topic.keywords)
        for video in topic.youtube_videos:
            update_rows.append([
                topic.name,
                keywords_string,
                video.id,
                f"https://www.youtube.com/watch?v={video.id}",
                video.title,
                video.description,
                video.published_at,
                video.channel_title,
                video.channel_id,
                video.thumbnail_url
            ])
    # update google sheet
    update_sheet_result = update_full_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = sheet_name_1,
        update_rows = update_rows
    )
    log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
    print(log_content)

    ####### 整理後影評影片ID
    # transform original data to table
    update_rows = [["Topic ", "Github action variable"]]
    for topic in topics:
        video_ids = [video.id for video in topic.youtube_videos]
        # 建立一個 Python 字典，包含 topic.name 和 video_ids
        github_action_variable = {
            "screenwork_name": topic.name,
            "video_ids": video_ids
        }     
        # 將字典轉換為 JSON 字串
        github_action_json_string = json.dumps(github_action_variable, ensure_ascii=False)   
        # 根據新的結構更新陣列
        update_rows.append([
            topic.name,
            github_action_json_string
        ])
    # update google sheet
    update_sheet_result = update_full_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = sheet_name_2,
        update_rows = update_rows
    )
    log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
    print(log_content)


finally:
    delete_secret_json()


# In[ ]:



