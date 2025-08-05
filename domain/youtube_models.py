from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class YoutubeVideo:
    id: str
    title: str
    description: str

@dataclass
class YoutubeComment:
    id: int
    parent_id: Optional[int]
    level: int
    textDisplay: str
    likeCount: int

@dataclass
class Topic:
    name: str
    keywords: List[str]
    youtube_videos: List[YoutubeVideo] = field(default_factory=list)

    def add_youtube_videos(self, youtube_videos: List[YoutubeVideo]):
        self.youtube_videos.extend(youtube_videos)
