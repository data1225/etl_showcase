from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class VideoSearchOrder(str, Enum):
    DATE = 'date'
    RATING = 'rating'
    RELEVANCE = 'relevance'
    TITLE = 'title'
    VIEW_COUNT = 'viewCount'

class CommentSearchStatus(str, Enum):
    Pending = 'Pending'
    Processing = 'Processing'
    Completed = 'Completed'

@dataclass
class YoutubeVideo:
    id: str
    title: str
    description: str
    published_at: str
    channel_title: str
    channel_id: str
    thumbnail_url: Optional[str] = None
    viewCount: int =  0
    likeCount: int =  0
    dislikeCount: int =  0
    commentCount: int =  0
    is_enable_comment: bool = False
    
@dataclass
class YoutubeComment:
    id: int
    video_id: int
    parent_id: Optional[int]
    level: int
    textDisplay: str
    likeCount: int
    published_at: str

@dataclass
class TopicDetail:
    keyword: str
    youtube_videos: List[YoutubeVideo] = field(default_factory=list)

    def add_youtube_videos(self, youtube_videos: List[YoutubeVideo]):
        self.youtube_videos.extend(youtube_videos)

    def remove_youtube_videos(self, remove_video_ids: List[str]):
        """
        從清單中移除指定video_id的影片。
        """
        
        # 使用列表生成式 (list comprehension) 來建立一個不包含要移除影片的新列表
        self.youtube_videos = [
            video for video in self.youtube_videos if video.id not in remove_video_ids
        ]

@dataclass
class Topic:
    name: str
    details: List[TopicDetail]

@dataclass
class CommentSearchState:
    screenwork: str
    status: CommentSearchStatus
    rest_video_ids: List[str]
    next_page_token: str
    log_time: datetime
