import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Optional
from etl_showcase.config.youtube import (
    YOUTUBE_API_KEY, 
    YOUTUBE_SPREADSHEET_ID, 
)
from etl_showcase.domain.models import BaseResponse, StatusCode
from etl_showcase.domain.youtube_models import (
    YoutubeVideo,
    YoutubeComment,
)
from etl_showcase.infrastructure.utils.time_utils import get_now_time_string

def get_youtube_service():
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def youtube_search_videos(
    query: str,               # 搜尋關鍵詞
    search_count: int,        # 欲撈取影片數，可視分析需求決定
    start_utc_datetime: str,  # 發佈起始時間
    end_utc_datetime: str     # 發佈截止時間
):
    service = get_youtube_service()
    videos: List[YoutubeVideo] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''
    next_page_token = None 
    current_page = 1 
    
    while True:
        try:
            search_response = service.search().list(
                part='snippet',
                maxResults=50, # API本身搜尋上限為50筆
                order='relevance',
                pageToken=next_page_token ,
                publishedAfter=start_utc_datetime,
                publishedBefore=end_utc_datetime,
                q=query,
                regionCode='TW',
                relevanceLanguage='zh',
                type='video'
            ).execute()
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
    video_ids: List[str],                          # 要抓取留言的 YouTube 影片 ID 列表
    max_comment_count_per_page: int = 100,         # 每頁抓取的留言數量上限
    max_page: Optional[int] = None,                # 僅在 "limit_max_page" 模式下使用，限制最大頁數
    current_next_page_token: Optional[str] = None, # 僅在 "auto" 模式下使用，續抓的起始 pageToken
    mode: str = "limit_max_page"                   # 模式: "limit_max_page", "full", "auto"
):
    # 避免 Python 預設參數的可變物件陷阱
    if video_ids is None:
        video_ids = []
    
    service = get_youtube_service()
    comments: List[YoutubeComment] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''

    for video_id in video_ids:
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
                    order='time',  # relevance , time
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
                    parent_id=None,
                    level=1,
                    textDisplay=top_comment['textDisplay'],
                    likeCount=top_comment['likeCount']
                ))
                comment_index += 1

                if 'replies' in item:
                    top_comment_id = comment_index - 1
                    for reply in item['replies']['comments']:
                        comments.append(YoutubeComment(
                            id=comment_index,
                            parent_id=top_comment_id,
                            level=2,
                            textDisplay=reply['snippet']['textDisplay'],
                            likeCount=reply['snippet']['likeCount']
                        ))
                        comment_index += 1

            # 更新下一頁 token
            next_page_token = response.get('nextPageToken')

            # 停止條件
            if not next_page_token:  # 沒有下一頁
                break
            if mode == "limit_max_page" and max_page is not None and current_page >= max_page:
                break

            current_page += 1

    return BaseResponse[YoutubeComment](
        status_code=result_status,
        message=f'Successfully got {len(comments)} youtube comments' if result_status == StatusCode.SUCCESS else result_message,
        content=comments
    )