import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Optional
from etl_showcase.config.youtube import (
    YOUTUBE_API_KEY, 
    YOUTUBE_SPREADSHEET_ID, 
)
from etl_showcase.domain.models import (
    BaseResponse, 
    StatusCode,
    RegionCode, 
    LanguageCode
)
from etl_showcase.domain.youtube_models import (
    YoutubeVideo,
    YoutubeComment,
    VideoSearchOrder,
)
from etl_showcase.infrastructure.utils.time_utils import get_now_time_string

def get_youtube_service():
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def youtube_search_videos(
    query: str,                                           # 搜尋關鍵詞
    search_count: int = 5,                                # 欲撈取影片總筆數
    start_utc_datetime: str = '',                         # 發佈起始時間
    end_utc_datetime: str = '',                           # 發佈截止時間
    order: VideoSearchOrder = VideoSearchOrder.RELEVANCE, # 搜尋排序
    regionCode: Optional[RegionCode] = None,              # 影片地區代碼
    relevanceLanguage: Optional[LanguageCode] = None,     # 影片語言代碼
):
    service = get_youtube_service()
    videos: List[YoutubeVideo] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''
    next_page_token = None 
    current_page = 1 
    
    while True:
        try:
            # 建立基本參數字典
            # youtube API搜尋上限為50筆，所以讓maxResults不超過50
            search_params = {
                'part': 'snippet',
                'maxResults': min(search_count, 50),
                'order': order.value,
                'pageToken': next_page_token,
                'q': query,
                'type': 'video'
            }
            
            # 根據條件動態加入參數
            if start_utc_datetime:
                search_params['publishedAfter'] = start_utc_datetime
            if end_utc_datetime:
                search_params['publishedBefore'] = end_utc_datetime
            if regionCode:
                search_params['regionCode'] = regionCode.value
            if relevanceLanguage:
                search_params['relevanceLanguage'] = relevanceLanguage.value
            
            # 使用字典解構傳入參數
            search_response = service.search().list(**search_params).execute()
        except HttpError as err:
            error_status = err.resp.status
            error_content = err.content.decode("utf-8") if hasattr(err.content, 'decode') else str(err.content)
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'[final next_page_token:{next_page_token}] HTTP {error_status}, {error_content}'
            break
        except Exception as err:
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'[final next_page_token:{next_page_token}] Unexpected error: {err}'
            break 
            
        new_videos = [
            YoutubeVideo(
                id=item['id']['videoId'],
                title=item['snippet']['title'],
                description=item['snippet']['description'],
                published_at=item['snippet']['publishedAt'],
                channel_title=item['snippet']['channelTitle'],
                channel_id=item['snippet']['channelId'],
                thumbnail_url=(
                    item.get('snippet', {})
                    .get('thumbnails', {})
                    .get('high', {})
                    .get('url')
                    or item.get('snippet', {})
                    .get('thumbnails', {})
                    .get('medium', {})
                    .get('url')
                    or item.get('snippet', {})
                    .get('thumbnails', {})
                    .get('default', {})
                    .get('url')
                )
            )
            for item in search_response.get('items', [])
            if item['id']['kind'] == 'youtube#video'
        ]
        videos.extend(new_videos)

        next_page_token = search_response.get('nextPageToken')
        search_count -= len(new_videos)
        if not next_page_token:
            result_status = StatusCode.SUCCESS
            break
        if search_count is not None and search_count <= 0:
            result_status = StatusCode.SUCCESS
            break      

    return BaseResponse[YoutubeVideo](
        status_code = result_status,
        message=f'Successfully got {len(videos)} youtube videos' if result_status == StatusCode.SUCCESS else result_message,
        content = videos
    )
    
def youtube_search_comments(
    video_id: str,                                 # 要抓取留言的 YouTube 影片 ID
    max_comment_count_per_page: int = 100,         # 限制每頁抓取留言數上限
    max_page: Optional[int] = None,                # 限制抓取最大頁數
    current_next_page_token: Optional[str] = None, # 僅在 "auto" 模式下使用，續抓的起始 pageToken
    mode: str = "auto"                             # 模式: "full", "auto"
):
    service = get_youtube_service()
    comments: List[YoutubeComment] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''

    comment_index = 1
    # 依模式決定起始 pageToken
    if mode == "auto":
        next_page_token = current_next_page_token
    else:
        next_page_token = None
    current_page = 1

    while True:
        try:
            response = service.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=max_comment_count_per_page,
                moderationStatus='published',
                order='relevance',  # relevance , time
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()
            result_status = StatusCode.SUCCESS
        except HttpError as err:
            error_status = err.resp.status
            error_content = err.content.decode("utf-8") if hasattr(err.content, 'decode') else str(err.content)
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'[final next_page_token:{next_page_token}][final video id:{video_id}] HTTP {error_status}, {error_content}'
            break
        except Exception as e:
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'[final next_page_token:{next_page_token}][final video id:{video_id}] Unexpected error: {e}'
            break

        # 處理留言與回覆
        for item in response.get('items', []):
            top_comment = item['snippet']['topLevelComment']['snippet']
            comments.append(YoutubeComment(
                id=comment_index,
                video_id = video_id,
                parent_id=None,
                level=1,
                textDisplay=top_comment['textDisplay'],
                likeCount=top_comment['likeCount'],
                published_at=top_comment['publishedAt'],
            ))
            comment_index += 1

            if 'replies' in item:
                top_comment_id = comment_index - 1
                for reply in item['replies']['comments']:
                    comments.append(YoutubeComment(
                        id=comment_index,
                        video_id = video_id,
                        parent_id=top_comment_id,
                        level=2,
                        textDisplay=reply['snippet']['textDisplay'],
                        likeCount=reply['snippet']['likeCount'],
                        published_at=reply['snippet']['publishedAt'],
                    ))
                    comment_index += 1

        # 更新下一頁 token
        next_page_token = response.get('nextPageToken')

        # 停止條件
        if not next_page_token:  # 沒有下一頁
            break
        if max_page is not None and current_page >= max_page:
            break

        current_page += 1

    return BaseResponse[YoutubeComment](
        status_code=result_status,
        message=f'Successfully got {len(comments)} youtube comments (include replies)' if result_status == StatusCode.SUCCESS else result_message,
        content=comments
    )