import json, os, time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Optional, Dict, Union
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

# local model
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

    return BaseResponse[List[YoutubeVideo]](
        status_code = result_status,
        message=f'Successfully got {len(videos)} youtube videos' if result_status == StatusCode.SUCCESS else result_message,
        content = videos
    )

def fetch_videos_by_ids(video_ids: List[str]) -> BaseResponse[YoutubeVideo]:
    service = get_youtube_service()
    videos: List[YoutubeVideo] = []
    result_status = StatusCode.WAIT_FOR_PROCESS
    result_message = ''
    
    # 將影片 ID 列表分批處理，每批次最多 50 個, range(start, stop, step) 
    for i in range(0, len(video_ids), 50):
        # 獲取當前批次的影片 ID
        current_batch = video_ids[i:i + 50]
        
        # 將 ID 列表轉換為以逗號分隔的字串
        video_ids_string = ",".join(current_batch)
        
        # 檢查是否還有 ID 需要處理
        if not video_ids_string:
            break
            
        try:
            # 建立參數字典，只包含當前批次的 ID
            search_params = {
                'part': 'snippet,statistics',
                'id': video_ids_string,
                'maxResults': 50, 
            }
            
            # 使用字典解構傳入參數
            search_response = service.videos().list(**search_params).execute()
        except HttpError as err:
            error_status = err.resp.status
            error_content = err.content.decode("utf-8") if hasattr(err.content, 'decode') else str(err.content)
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'HTTP {error_status}, {error_content}'
            break
        except Exception as err:
            result_status = StatusCode.CALL_API_FAIL
            result_message = f'Unexpected error: {err}'
            break
            
        new_videos = [
            YoutubeVideo(
                id=item['id'],
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
                ),
                viewCount=item['statistics'].get('viewCount', 0),
                likeCount=item['statistics'].get('likeCount', 0),
                dislikeCount=item['statistics'].get('dislikeCount', 0),
                commentCount=item['statistics'].get('commentCount', 0),
                is_enable_comment='commentCount' in item['statistics']
            )
            for item in search_response.get('items', [])
        ]
        videos.extend(new_videos)

    # 迴圈結束後，檢查是否所有影片都已成功獲取
    if not result_message:
        result_status = StatusCode.SUCCESS
    
    return BaseResponse[List[YoutubeVideo]](
        status_code=result_status,
        message=f'Successfully got {len(videos)} videos info by ids' if result_status == StatusCode.SUCCESS else result_message,
        content=videos
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

def fetch_video_cc_subtitle(video_id: str, languages: List[str], retry_count: int = 3) -> Optional[str]:
    """
    透過 YouTube 影片 ID 取得影片內容，並加入重試機制與代理伺服器支援。
    
    Args:
        video_id (str): YouTube 影片的 ID。
        languages (List[str]): 偏好的語言代碼列表。
        retry_count (int): 發生錯誤時的重試次數。
    
    Returns:
        Optional[str]: 影片的文字內容，若無法取得則返回 None。
    """
    transcript_text = ""

    print(f"嘗試取得影片 ID: {video_id} 的官方字幕...")
    
    for attempt in range(retry_count):
        try:
            # 嘗試取得CC字幕，並將代理伺服器傳入
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(
                video_id, 
                languages=languages
            )           
            # 使用 .text 屬性來存取字幕內容
            transcript_text = " ".join([snippet['text'] for snippet in transcript_list])
            print("✅ 成功取得官方字幕。")
            return transcript_text

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"找不到影片 ID: {video_id} 的字幕，回傳空白。")
            return ""
        
        except Exception as e:
            print(f"第 {attempt + 1}/{retry_count} 次嘗試發生未知錯誤：{e}")
            
            if attempt < retry_count - 1:
                # 如果還有重試次數，則暫停一段時間後再試
                delay = 2 ** attempt  # 這裡使用指數退避策略
                print(f"等待 {delay} 秒後重試...")
                time.sleep(delay)
            else:
                # 所有重試都失敗，結束
                print("所有重試都失敗。")
                return None
    
    return None


def fetch_video_content(video_id: str, languages: list[str]) -> Optional[str]:
    """
    透過 YouTube 影片 ID 取得影片內容。
    優先嘗試取得官方字幕，若無則將語音轉文字。
    
    Args:
        video_id (str): YouTube 影片的 ID。
        languages (list[str]): 偏好的語言代碼列表。
    
    Returns:
        Optional[str]: 影片的文字內容，若無法取得則返回 None。
    """
    transcript_text = ""

    print(f"嘗試取得影片 ID: {video_id} 的官方字幕...")
    try:
        # 優先嘗試取得CC字幕
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(
            video_id, 
            languages=languages
        )
        
        # 將字幕片段合併成一個完整的字串
        transcript_text = " ".join([snippet['text'] for snippet in transcript_list])
        print("✅ 成功取得官方字幕。")
        return transcript_text

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"影片沒有官方字幕")
        print("轉為使用語音轉文字...")

        # 根據 download_path 建立音頻檔案路徑
        download_dir = "./data/audio"
        os.makedirs(os.path.dirname(download_dir), exist_ok=True)
        output_template = os.path.join(download_dir, video_id)
        audio_file_path = f'{output_template}.mp3'

        if not os.path.exists(audio_file_path):
            # 使用 yt-dlp 下載音頻
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'noprogress': True,
            }
    
            try:
                print(f"🎧 開始下載影片音軌至：{audio_file_path}")
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            except Exception as e:
                print(f"下載音軌失敗：{e}")
                return None

        transcribed_text = None
        # 確保檔案存在，否則轉錄會失敗
        if not os.path.exists(audio_file_path):
            print("音軌檔案不存在，無法進行轉錄。")
            return None
        
        try:
            print("🔊 下載完成，開始語音轉文字...")
            
            # 載入 Faster-Whisper 模型，可依需求調整模型大小
            model_size = "base"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

            # 轉錄音頻
            segments, info = model.transcribe(audio_file_path, beam_size=5)
            
            transcribed_text = " ".join([segment.text for segment in segments])
            print("✅ 語音轉文字完成。")
            
        except Exception as e:
            print(f"發生語音轉文字錯誤：{e}")
        finally:
            # 無論成功與否，都刪除音頻檔案
            if os.path.exists(audio_file_path):
                print(f"🗑️ 正在刪除音頻檔案：{audio_file_path}")
                os.remove(audio_file_path)
                print("✅ 檔案刪除成功。")
            
        return transcribed_text

    except Exception as e:
        print(f"發生未知錯誤：{e}")
        return None