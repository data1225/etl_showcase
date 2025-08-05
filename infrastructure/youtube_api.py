import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Optional
from film_tv_animation_insights.config.youtube import (
    YOUTUBE_API_KEY, 
    YOUTUBE_SPREADSHEET_ID, 
)
from film_tv_animation_insights.domain.models import BaseResponse, StatusCode
from film_tv_animation_insights.domain.youtube_models import (
    YoutubeVideo,
    YoutubeComment,
)
from film_tv_animation_insights.infrastructure.time_utils import get_now_time_string

def get_youtube_service():
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def youtube_search_videos(query, start_utc_datetime, end_utc_datetime):
    service = get_youtube_service()
    try:
        search_response = service.search().list(
            part='snippet',
            maxResults=50,
            order='relevance',
            publishedAfter=start_utc_datetime,
            publishedBefore=end_utc_datetime,
            q=query,
            regionCode='TW',
            relevanceLanguage='zh',
            type='video'
        ).execute()
    except HttpError as e:
        error_status = e.resp.status
        error_content = e.content.decode("utf-8") if hasattr(e.content, 'decode') else str(e.content)
        return BaseResponse[YoutubeVideo](
            status_code=StatusCode.CALL_API_FAIL,
            message=f'[query:{query}] HTTP {error_status}, {error_content}',
            content=videos
        )

    videos = [
        YoutubeVideo(
            id=item['id']['videoId'],
            title=item['snippet']['title'],
            description=item['snippet']['description']
        ) for item in search_response.get('items', []) if item['id']['kind'] == 'youtube#video']
    return BaseResponse[YoutubeVideo](
        status_code=StatusCode.SUCCESS,
        message=f'[query:{query}] Successfully get {len(videos)} youtube videos',
        content=videos
    )

def youtube_search_comments(video_ids: List[str] = None, max_comment_count_per_page: int = 100, max_page: Optional[int] = None):
    # 避免python []全域共用一實例問題
    if video_ids is None:
        video_ids = []
    
    service = get_youtube_service()
    comments: List[YoutubeComment] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''
    for video_id in video_ids:        
        comment_index = 1
        next_page_token = None 
        current_page = 1 
        while True:
            try:        
                response = service.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=max_comment_count_per_page,
                    moderationStatus='published',
                    order='relevance',# relevance , time
                    textFormat='plainText',
                    pageToken=next_page_token
                ).execute()
                # print(json.dumps(response, indent=2, ensure_ascii=False))
                result_status = StatusCode.SUCCESS
            except HttpError as e:
                error_status = err.resp.status
                error_content = err.content.decode("utf-8") if hasattr(err.content, 'decode') else str(err.content)
                result_status = StatusCode.CALL_API_FAIL
                result_message = f'[final next_page_token:{next_page_token}][final video id:{video_id}] HTTP {error_status}, {error_content}'
                break
            except Exception as e:
                result_status = StatusCode.CALL_API_FAIL
                result_message = f'[final next_page_token:{next_page_token}][final video id:{video_id}] Unexpected error: {e}'
                break

            for item in response.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                comments.append(YoutubeComment(
                    id = comment_index,
                    parent_id = None,
                    level = 1,
                    textDisplay = top_comment['textDisplay'],
                    likeCount = top_comment['likeCount']
                ))
                if 'replies' in item:
                    top_comment_id = comment_index
                    comment_index + 1
                    for reply in item['replies']['comments']:
                        comments.append(YoutubeComment(
                            id = comment_index,
                            parent_id = top_comment_id,
                            level = 2,
                            textDisplay = reply['snippet']['textDisplay'],
                            likeCount = reply['snippet']['likeCount']
                        ))
                        # for next id of previous comment that has replies
                        comment_index + 1
                else:
                    # for next id of previous comment that don't has replies
                    comment_index + 1
                        
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            if max_page is not None and current_page >= max_page:
                break
            current_page + 1

    return BaseResponse[YoutubeComment](
        status_code = result_status,
        message=f'Successfully got {len(comments)} youtube comments' if result_status == StatusCode.SUCCESS else result_message,
        content = comments
    )