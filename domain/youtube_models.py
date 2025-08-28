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
