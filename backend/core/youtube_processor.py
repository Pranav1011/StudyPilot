"""
YouTube Video Processor

Core functionality for processing YouTube videos and playlists.
Downloads audio, extracts metadata, detects educational content, and manages caching.
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from urllib.parse import urlparse, parse_qs
import json
import hashlib

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

logger = logging.getLogger(__name__)


class YouTubeProcessorError(Exception):
    """Base exception for YouTube processor errors."""
    pass


class InvalidURLError(YouTubeProcessorError):
    """Raised when YouTube URL is invalid or unsupported."""
    pass


class DownloadError(YouTubeProcessorError):
    """Raised when video download fails."""
    pass


class ProcessingError(YouTubeProcessorError):
    """Raised when video processing fails."""
    pass


class AgeRestrictedError(YouTubeProcessorError):
    """Raised when video is age-restricted."""
    pass


class PrivateVideoError(YouTubeProcessorError):
    """Raised when video is private."""
    pass


class YouTubeProcessor:
    """
    YouTube video processor with comprehensive functionality.
    
    Features:
    - Audio-only downloads (MP3, 96kbps)
    - Metadata extraction
    - Educational content detection
    - Caching and storage management
    - Playlist support
    - Error handling and retries
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache/youtube",
        max_concurrent: int = 3,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
        max_cache_size_gb: int = 10
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.max_cache_size_gb = max_cache_size_gb
        
        # Educational keywords for content detection
        self.educational_keywords = {
            'academic': ['lecture', 'tutorial', 'course', 'lesson', 'education', 'learning'],
            'subjects': ['math', 'science', 'history', 'physics', 'chemistry', 'biology', 
                        'programming', 'computer science', 'engineering', 'economics'],
            'formats': ['how to', 'explained', 'introduction', 'basics', 'advanced', 
                       'concept', 'theory', 'practice', 'examples']
        }
        
        # URL patterns for validation
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
        ]
        
        logger.info(f"YouTubeProcessor initialized with cache_dir={cache_dir}")
    
    def extract_video_id(self, url: str) -> str:
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID
            
        Raises:
            InvalidURLError: If URL is not a valid YouTube URL
        """
        for pattern in self.youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise InvalidURLError(f"Invalid YouTube URL: {url}")
    
    def is_playlist_url(self, url: str) -> bool:
        """Check if URL is a playlist URL."""
        return 'playlist' in url or 'list=' in url
    
    def validate_url(self, url: str) -> bool:
        """
        Validate YouTube URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid YouTube URL
        """
        try:
            self.extract_video_id(url)
            return True
        except InvalidURLError:
            return False
    
    def _get_cache_key(self, video_id: str) -> str:
        """Generate cache key for video."""
        return hashlib.md5(video_id.encode()).hexdigest()
    
    def _get_cache_path(self, video_id: str) -> Path:
        """Get cache file path for video."""
        cache_key = self._get_cache_key(video_id)
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cached(self, video_id: str) -> bool:
        """Check if video is cached."""
        cache_path = self._get_cache_path(video_id)
        return cache_path.exists()
    
    def _load_cache(self, video_id: str) -> Optional[Dict]:
        """Load cached data for video."""
        try:
            cache_path = self._get_cache_path(video_id)
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded cached data for video {video_id}")
                    return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {video_id}: {e}")
        return None
    
    def _save_cache(self, video_id: str, data: Dict) -> None:
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(video_id)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached data for video {video_id}")
        except Exception as e:
            logger.error(f"Failed to save cache for {video_id}: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache files if size exceeds limit."""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            max_size = self.max_cache_size_gb * 1024 * 1024 * 1024
            
            if total_size > max_size:
                logger.info("Cache size exceeded limit, cleaning up old files...")
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.rglob('*.json')]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                # Remove oldest files until under limit
                for file_path, _ in files:
                    if total_size <= max_size:
                        break
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_size -= file_size
                    logger.info(f"Removed old cache file: {file_path}")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _detect_educational_content(self, metadata: Dict) -> Dict:
        """
        Detect if video content is educational.
        
        Args:
            metadata: Video metadata
            
        Returns:
            Dictionary with educational content analysis
        """
        title = metadata.get('title', '').lower()
        description = metadata.get('description', '').lower()
        tags = [tag.lower() for tag in metadata.get('tags', [])]
        category = metadata.get('category', '').lower()
        
        # Check for educational indicators
        educational_score = 0
        detected_subjects = []
        difficulty_level = 'beginner'
        
        # Title analysis
        for subject in self.educational_keywords['subjects']:
            if subject in title:
                educational_score += 2
                detected_subjects.append(subject)
        
        for format_type in self.educational_keywords['formats']:
            if format_type in title:
                educational_score += 1
        
        # Description analysis
        for subject in self.educational_keywords['subjects']:
            if subject in description:
                educational_score += 1
                if subject not in detected_subjects:
                    detected_subjects.append(subject)
        
        # Tags analysis
        for tag in tags:
            for subject in self.educational_keywords['subjects']:
                if subject in tag:
                    educational_score += 1
                    if subject not in detected_subjects:
                        detected_subjects.append(subject)
        
        # Category analysis
        if category in ['education', 'science & technology', 'howto & style']:
            educational_score += 2
        
        # Duration analysis (longer videos tend to be more educational)
        duration = metadata.get('duration', 0)
        if duration > 1800:  # 30 minutes
            educational_score += 1
            difficulty_level = 'intermediate'
        if duration > 3600:  # 1 hour
            educational_score += 1
            difficulty_level = 'advanced'
        
        # Determine if educational
        is_educational = educational_score >= 3
        
        return {
            'is_educational': is_educational,
            'educational_score': educational_score,
            'detected_subjects': list(set(detected_subjects)),
            'difficulty_level': difficulty_level,
            'analysis_factors': {
                'title_score': sum(1 for subject in self.educational_keywords['subjects'] if subject in title),
                'description_score': sum(1 for subject in self.educational_keywords['subjects'] if subject in description),
                'tags_score': sum(1 for tag in tags for subject in self.educational_keywords['subjects'] if subject in tag),
                'category_score': 2 if category in ['education', 'science & technology', 'howto & style'] else 0,
                'duration_score': 1 if duration > 1800 else 0
            }
        }
    
    def _get_yt_dlp_options(self, output_path: str, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Get yt-dlp options for audio download.
        
        Args:
            output_path: Output file path
            progress_callback: Optional progress callback
            
        Returns:
            yt-dlp options dictionary
        """
        options = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '96',
            }],
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'ignoreerrors': False,
            'no_warnings': False,
            'quiet': False,
            'extract_flat': False,
        }
        
        if progress_callback:
            options['progress_hooks'] = [progress_callback]
        
        return options
    
    def _download_with_retry(
        self, 
        url: str, 
        output_path: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Download video with retry logic.
        
        Args:
            url: YouTube URL
            output_path: Output file path
            progress_callback: Optional progress callback
            
        Returns:
            Download result dictionary
            
        Raises:
            DownloadError: If download fails after retries
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1} for {url}")
                
                # Add delay between retries
                if attempt > 0:
                    delay = self.rate_limit_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
                
                options = self._get_yt_dlp_options(output_path, progress_callback)
                
                with yt_dlp.YoutubeDL(options) as ydl:
                    # First, extract info without downloading
                    info = ydl.extract_info(url, download=False)
                    
                    # Then download
                    ydl.download([url])
                
                # Verify download
                if not os.path.exists(output_path):
                    raise DownloadError("Download completed but file not found")
                
                return info
                
            except DownloadError as e:
                last_error = e
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                
                # Clean up partial download
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                
                # Check for specific error types
                if "age-restricted" in str(e).lower():
                    raise AgeRestrictedError(f"Age-restricted video: {url}")
                elif "private" in str(e).lower():
                    raise PrivateVideoError(f"Private video: {url}")
                
            except ExtractorError as e:
                last_error = e
                logger.warning(f"Extractor error on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        raise DownloadError(f"Download failed after {self.max_retries} attempts: {last_error}")
    
    def _extract_metadata(self, info: Dict) -> Dict:
        """
        Extract comprehensive metadata from video info.
        
        Args:
            info: yt-dlp info dictionary
            
        Returns:
            Processed metadata dictionary
        """
        metadata = {
            'video_id': info.get('id'),
            'title': info.get('title'),
            'channel': info.get('uploader'),
            'channel_id': info.get('channel_id'),
            'duration': info.get('duration'),
            'upload_date': info.get('upload_date'),
            'description': info.get('description'),
            'tags': info.get('tags', []),
            'category': info.get('category'),
            'view_count': info.get('view_count'),
            'like_count': info.get('like_count'),
            'thumbnail_url': info.get('thumbnail'),
            'webpage_url': info.get('webpage_url'),
            'is_live': info.get('is_live', False),
            'was_live': info.get('was_live', False),
            'live_status': info.get('live_status'),
            'availability': info.get('availability'),
            'chapters': info.get('chapters', []),
            'playlist_index': info.get('playlist_index'),
            'playlist_id': info.get('playlist_id'),
            'playlist_title': info.get('playlist_title'),
            'automatic_captions': info.get('automatic_captions', {}),
            'subtitles': info.get('subtitles', {}),
        }
        
        # Add educational content detection
        educational_analysis = self._detect_educational_content(metadata)
        metadata['educational_analysis'] = educational_analysis
        
        return metadata
    
    async def process_video(
        self, 
        url: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process a single YouTube video.
        
        Args:
            url: YouTube video URL
            progress_callback: Optional progress callback function
            
        Returns:
            Processing result dictionary
            
        Raises:
            InvalidURLError: If URL is invalid
            DownloadError: If download fails
            ProcessingError: If processing fails
        """
        try:
            # Validate URL
            if not self.validate_url(url):
                raise InvalidURLError(f"Invalid YouTube URL: {url}")
            
            video_id = self.extract_video_id(url)
            logger.info(f"Processing video: {video_id}")
            
            # Check cache first
            if self._is_cached(video_id):
                cached_data = self._load_cache(video_id)
                if cached_data:
                    logger.info(f"Using cached data for video {video_id}")
                    return cached_data
            
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"{video_id}.mp3")
                
                # Download video
                info = self._download_with_retry(url, output_path, progress_callback)
                
                # Extract metadata
                metadata = self._extract_metadata(info)
                
                # Move file to cache directory
                final_path = self.cache_dir / f"{video_id}.mp3"
                shutil.move(output_path, final_path)
                
                # Prepare result
                result = {
                    'video_id': video_id,
                    'url': url,
                    'audio_path': str(final_path),
                    'metadata': metadata,
                    'processing_time': time.time(),
                    'cache_hit': False
                }
                
                # Save to cache
                self._save_cache(video_id, result)
                
                # Cleanup old cache if needed
                self._cleanup_cache()
                
                logger.info(f"Successfully processed video {video_id}")
                return result
                
        except (InvalidURLError, DownloadError, AgeRestrictedError, PrivateVideoError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing video {url}: {e}")
            raise ProcessingError(f"Failed to process video: {e}")
    
    async def process_playlist(
        self, 
        playlist_url: str, 
        progress_callback: Optional[Callable] = None,
        max_videos: Optional[int] = None
    ) -> List[Dict]:
        """
        Process a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            progress_callback: Optional progress callback function
            max_videos: Maximum number of videos to process
            
        Returns:
            List of processing results
            
        Raises:
            InvalidURLError: If URL is invalid
            ProcessingError: If processing fails
        """
        try:
            if not self.is_playlist_url(playlist_url):
                raise InvalidURLError(f"Not a playlist URL: {playlist_url}")
            
            logger.info(f"Processing playlist: {playlist_url}")
            
            # Extract playlist info
            options = {
                'extract_flat': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(options) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if not playlist_info or 'entries' not in playlist_info:
                raise ProcessingError("Failed to extract playlist information")
            
            videos = playlist_info['entries']
            if max_videos:
                videos = videos[:max_videos]
            
            logger.info(f"Found {len(videos)} videos in playlist")
            
            # Process videos concurrently
            results = []
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_video_with_semaphore(video_info):
                async with semaphore:
                    try:
                        video_url = video_info.get('url')
                        if video_url:
                            return await self.process_video(video_url, progress_callback)
                    except Exception as e:
                        logger.error(f"Failed to process video in playlist: {e}")
                        return None
            
            # Create tasks for all videos
            tasks = [process_video_with_semaphore(video) for video in videos]
            
            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            for result in completed_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                elif result is not None:
                    results.append(result)
            
            logger.info(f"Successfully processed {len(results)} videos from playlist")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process playlist {playlist_url}: {e}")
            raise ProcessingError(f"Playlist processing failed: {e}")
    
    def get_processing_status(self, video_id: str) -> Dict:
        """
        Get processing status for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Status dictionary
        """
        cache_path = self._get_cache_path(video_id)
        audio_path = self.cache_dir / f"{video_id}.mp3"
        
        return {
            'video_id': video_id,
            'is_cached': cache_path.exists(),
            'has_audio': audio_path.exists(),
            'cache_size': cache_path.stat().st_size if cache_path.exists() else 0,
            'audio_size': audio_path.stat().st_size if audio_path.exists() else 0,
        }
    
    def cleanup_video(self, video_id: str) -> bool:
        """
        Clean up cached data for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if cleanup successful
        """
        try:
            cache_path = self._get_cache_path(video_id)
            audio_path = self.cache_dir / f"{video_id}.mp3"
            
            if cache_path.exists():
                cache_path.unlink()
            
            if audio_path.exists():
                audio_path.unlink()
            
            logger.info(f"Cleaned up cached data for video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup video {video_id}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        try:
            cache_files = list(self.cache_dir.rglob('*.json'))
            audio_files = list(self.cache_dir.rglob('*.mp3'))
            
            total_cache_size = sum(f.stat().st_size for f in cache_files)
            total_audio_size = sum(f.stat().st_size for f in audio_files)
            
            return {
                'cache_files_count': len(cache_files),
                'audio_files_count': len(audio_files),
                'total_cache_size_bytes': total_cache_size,
                'total_audio_size_bytes': total_audio_size,
                'total_cache_size_mb': total_cache_size / (1024 * 1024),
                'total_audio_size_mb': total_audio_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {} 