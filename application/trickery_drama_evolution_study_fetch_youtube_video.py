#!/usr/bin/env python
# coding: utf-8

# In[7]:


#################################################################
# 男頻高流量權謀爽劇演進研究：搜集熱門影評及熱門留言
#################################################################

    
from path_setup import setup_project_root
root = setup_project_root()

import dataclasses, json, time
from etl_showcase.domain.models import (
    BaseResponse, 
    StatusCode,
    LanguageCode,
)
from etl_showcase.domain.youtube_models import (
    YoutubeVideo, 
    Topic,
    TopicDetail,
    VideoSearchOrder,
)
from etl_showcase.infrastructure.utils.time_utils import get_previous_month_range_in_utc
from etl_showcase.infrastructure.datasource.youtube_api import (
    youtube_search_videos,
    fetch_videos_by_ids,
)
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    update_full_google_sheet,
    create_google_sheet,
    is_sheet_exists,
)
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
)

# 宣告一個繼承 Topic 的新 class
@dataclasses.dataclass
class ScreenworkTopic(Topic):
    cultural_sphere: str = 'Default'
    category: str = 'Default'
    publish_year: int = 0

# 使用新宣告的 class 並填入預設值
topics = [ 
    ScreenworkTopic('琅琊榜', [TopicDetail('琅琊榜 影評')], 'zh', '復仇劇', 2015),
    ScreenworkTopic('Nirvana in Fire', [TopicDetail('Nirvana in Fire review')], 'en', '復仇劇', 2015),    
    ScreenworkTopic('Joy of Life Season 1', [TopicDetail('Joy of Life Season 1 review')], 'en', '一般權謀劇', 2015),       
    ScreenworkTopic('My Heroic Husband', [TopicDetail('My Heroic Husband review')], 'en', '一般權謀劇', 2021),
    ScreenworkTopic('雪中悍刀行', [TopicDetail('雪中悍刀行 影評')], 'zh', '一般權謀劇 ', 2021),
    ScreenworkTopic('Sword Snow Stride', [TopicDetail('Sword Snow Stride review')], 'en', '一般權謀劇', 2021),
    ScreenworkTopic('慶餘年 第二季', [TopicDetail('慶餘年 影評')], 'zh', '一般權謀劇', 2024),
    ScreenworkTopic('Joy of Life Season 2', [TopicDetail('Joy of Life Season 2 review')], 'en', '一般權謀劇', 2024),
    ScreenworkTopic('藏海傳', [TopicDetail('藏海傳 影評')], 'zh', '復仇劇 ', 2025),
    ScreenworkTopic('The Legend of Zang Hai', [TopicDetail('The Legend of Zang Hai review')], 'en', '復仇劇', 2025),
    # 慶餘年第一季、贅婿 其搜尋關鍵字如只放「劇名 影評」會撈到許多不相關影片，故把搜尋關鍵字改成「劇名 電視劇 影評」再重新撈取資料。
    ScreenworkTopic('慶餘年 第一季', [TopicDetail('慶餘年 電視劇 影評')], 'zh', '一般權謀劇 ', 2019),
    ScreenworkTopic('贅婿', [TopicDetail('贅婿 電視劇 影評')], 'zh', '一般權謀劇', 2021), 
]

print('開始搜尋影片資料')
search_youtube_result = BaseResponse[YoutubeVideo](
    status_code=StatusCode.WAIT_FOR_PROCESS,
    message='',
    content=None
)    
for topic in topics:
    for detail in topic.details:
        # 根據 cultural_sphere 判斷語言
        if topic.cultural_sphere == 'en':
            relevanceLanguage = LanguageCode.English
        else:
            relevanceLanguage = LanguageCode.Chinese

        start_utc_datetime = ''
        end_utc_datetime = ''
        # 因一二季的影評混在一起，所以用時間分開搜尋影評。
        # 第二季首播是2024年5月16日，但可能是為了宣傳第二季或頻道自身經營流量需求，
        # 2023年就有慶餘年第二季資料，故把區隔時間訂在+8時區2023年1月1日。
        if topic.name == '慶餘年 第一季':
            end_utc_datetime = '2022-12-31T15:59:59Z'
        elif topic.name == '慶餘年 第二季':
            start_utc_datetime = '2022-12-31T16:00:00Z'

        search_youtube_result = youtube_search_videos(
            query=detail.keyword,
            search_count=60,
            start_utc_datetime = start_utc_datetime,
            end_utc_datetime = end_utc_datetime,
            order=VideoSearchOrder.RELEVANCE,
            relevanceLanguage=relevanceLanguage
        )
        print(f'keyword: {detail.keyword}, result: [{search_youtube_result.status_code}] {search_youtube_result.message}')
        if search_youtube_result.content is None:
            break
        detail.add_youtube_videos(search_youtube_result.content)

print('\r\n開始搜尋影片按讚數，並過濾禁止留言功能影片')
video_likes = {}
for topic in topics:  
    for detail in topic.details:
        detail_video_ids = []
        for video in detail.youtube_videos:
            detail_video_ids.append(video.id)
            
        fetch_videos_by_ids_result = fetch_videos_by_ids(detail_video_ids)
        print(f'fetch videos result: [{fetch_videos_by_ids_result.status_code}] {fetch_videos_by_ids_result.message}')
        
        remove_video_ids = []
        new_video_likes=[]        
        if fetch_videos_by_ids_result.content is not None:
            new_video_likes = {video.id: video.likeCount for video in fetch_videos_by_ids_result.content}
            remove_video_ids = [video.id for video in fetch_videos_by_ids_result.content if video.is_enable_comment is not True]
            detail.remove_youtube_videos(remove_video_ids)

        video_likes.update(new_video_likes)
        remove_video_ids_count = len(remove_video_ids)
        print(f'remove_video_ids len:{len(remove_video_ids)}')
        if remove_video_ids_count > 0:
            print(f'作品{topic.name}刪除禁止無留言影片{remove_video_ids_count}筆，內容為{remove_video_ids}')  
        else:
            print(f'作品{topic.name}無禁止無留言影片，保留原搜尋影片。')  

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
    print('\r\n記影評影片資料')
    # transform original data to table
    print('\r\ntransform original data to table')
    update_rows = [["Screenwork", "Publish year", "Cultural sphere", "Category", "Search keyword", 
                     "Video ID", "Video URL", "Video title", "Video description", "Like count"]]
    for topic in topics:
        for detail in topic.details:
            for video in detail.youtube_videos:
                update_rows.append([
                    topic.name,
                    topic.publish_year,
                    topic.cultural_sphere,
                    topic.category,
                    detail.keyword,
                    video.id,
                    f"https://www.youtube.com/watch?v={video.id}",
                    video.title,
                    video.description,
                    video_likes.get(video.id, 0),
                ])
    # update google sheet
    print('\r\nupdate google sheet')
    update_sheet_result = update_full_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = sheet_name_1,
        update_rows = update_rows
    )
    log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
    print(f'\r\n{log_content}')

    ####### 整理後讀取留言用的 影評影片ID json內容
    print('\r\n整理後讀取留言用的 影評影片ID json內容')
    # transform original data to table
    print('\r\ntransform original data to table')
    update_rows = [["Topic ", "Github action variable"]]
    for topic in topics:
        for detail in topic.details:
            video_ids = [video.id for video in detail.youtube_videos]
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
    print('\r\nupdate google sheet')
    update_sheet_result = update_full_google_sheet(
        spreadsheet_id = YOUTUBE_SPREADSHEET_ID,
        sheet_name = sheet_name_2,
        update_rows = update_rows
    )
    log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
    print(f'\r\n{log_content}')


finally:
    delete_secret_json()


# In[ ]:



