"""
Transcription Service

Core functionality for transcribing educational content with Whisper,
detecting educational patterns, and creating smart segments.
"""

import asyncio
import logging
import os
import re
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import tempfile
import threading
from contextlib import contextmanager

import whisper
import numpy as np
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class AudioFileError(TranscriptionError):
    """Raised when audio file is corrupted or invalid."""
    pass


class TranscriptionTimeoutError(TranscriptionError):
    """Raised when transcription times out."""
    pass


class MemoryError(TranscriptionError):
    """Raised when insufficient memory for transcription."""
    pass


@dataclass
class WordSegment:
    """Represents a word with timestamp and confidence."""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptSegment:
    """Represents a smart segment with educational metadata."""
    id: str
    start: float
    end: float
    text: str
    segment_type: str  # 'definition', 'example', 'important', 'transition', 'summary', 'content'
    importance_score: float  # 0.0 to 1.0
    key_terms: List[str]
    confidence: float
    word_count: int
    duration: float


@dataclass
class KeyLearningMoment:
    """Represents a key learning moment with context."""
    timestamp: float
    type: str  # 'definition', 'example', 'important_point', 'summary'
    text: str
    importance_score: float
    related_segments: List[str]  # segment IDs
    context: str


@dataclass
class TranscriptionResult:
    """Complete transcription result with all metadata."""
    full_text: str
    segments: List[TranscriptSegment]
    key_moments: List[KeyLearningMoment]
    language: str
    language_confidence: float
    total_duration: float
    word_count: int
    average_confidence: float
    processing_time: float
    cache_hit: bool


class EducationalTranscriptionService:
    """
    Educational content transcription service with Whisper integration.
    
    Features:
    - Word-level transcription with timestamps
    - Educational pattern detection
    - Smart segmentation
    - Key learning moment extraction
    - Performance optimizations
    """
    
    def __init__(
        self,
        model_name: str = "base",
        cache_dir: str = "./cache/transcriptions",
        max_workers: int = 4,
        chunk_duration: int = 300,  # 5 minutes
        timeout_seconds: int = 1800,  # 30 minutes
        max_memory_gb: int = 8
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.chunk_duration = chunk_duration
        self.timeout_seconds = timeout_seconds
        self.max_memory_gb = max_memory_gb
        
        # Educational pattern detection
        self.educational_patterns = {
            'definitions': [
                r'\b(?:is defined as|means|we call this|refers to|denotes|represents)\b',
                r'\b(?:definition of|what is|define)\b',
                r'\b(?:in other words|that is|i\.e\.|namely)\b'
            ],
            'examples': [
                r'\b(?:for example|for instance|consider|take|suppose)\b',
                r'\b(?:such as|like|including|e\.g\.)\b',
                r'\b(?:imagine|picture|think of)\b'
            ],
            'important_points': [
                r'\b(?:key point|important|remember|note that|crucial)\b',
                r'\b(?:pay attention|focus on|highlight|emphasize)\b',
                r'\b(?:this is critical|essential|vital|significant)\b'
            ],
            'transitions': [
                r'\b(?:moving on|next|now let\'s|turning to|shifting to)\b',
                r'\b(?:meanwhile|however|on the other hand|in contrast)\b',
                r'\b(?:furthermore|moreover|additionally|also)\b'
            ],
            'summaries': [
                r'\b(?:to recap|in summary|key takeaways|to summarize)\b',
                r'\b(?:in conclusion|finally|overall|in brief)\b',
                r'\b(?:the main point|the bottom line|the gist)\b'
            ]
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for pattern_type, patterns in self.educational_patterns.items():
            self.compiled_patterns[pattern_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        # Load Whisper model
        self._load_model()
        
        logger.info(f"EducationalTranscriptionService initialized with model={model_name}")
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.tokenizer = get_tokenizer(self.model.is_multilingual)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise TranscriptionError(f"Model loading failed: {e}")
    
    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key for audio file."""
        file_hash = hashlib.md5()
        with open(audio_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if transcription is cached."""
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached transcription."""
        try:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded cached transcription: {cache_key}")
                    return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
        return None
    
    def _save_cache(self, cache_key: str, data: Dict) -> None:
        """Save transcription to cache."""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Cached transcription: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to save cache for {cache_key}: {e}")
    
    def _detect_educational_patterns(self, text: str, start_time: float, end_time: float) -> Dict:
        """
        Detect educational patterns in text segment.
        
        Args:
            text: Text to analyze
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Pattern detection results
        """
        text_lower = text.lower()
        detected_patterns = {}
        importance_score = 0.0
        key_terms = []
        
        # Check each pattern type
        for pattern_type, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                for match in pattern.finditer(text_lower):
                    matches.append({
                        'pattern': pattern.pattern,
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group()
                    })
            
            if matches:
                detected_patterns[pattern_type] = matches
                
                # Calculate importance based on pattern type
                if pattern_type == 'definitions':
                    importance_score += 0.8
                elif pattern_type == 'important_points':
                    importance_score += 0.9
                elif pattern_type == 'examples':
                    importance_score += 0.6
                elif pattern_type == 'summaries':
                    importance_score += 0.7
                elif pattern_type == 'transitions':
                    importance_score += 0.3
        
        # Extract potential key terms (capitalized words, technical terms)
        key_terms = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
        key_terms.extend(re.findall(r'\b[a-z]+(?:\s+[a-z]+)*\b', text))
        
        # Normalize importance score
        importance_score = min(importance_score, 1.0)
        
        return {
            'patterns': detected_patterns,
            'importance_score': importance_score,
            'key_terms': list(set(key_terms))[:10],  # Limit to 10 terms
            'pattern_count': sum(len(matches) for matches in detected_patterns.values())
        }
    
    def _create_smart_segments(
        self, 
        words: List[WordSegment], 
        target_duration: Tuple[float, float] = (30.0, 90.0)
    ) -> List[TranscriptSegment]:
        """
        Create smart segments based on educational structure.
        
        Args:
            words: List of words with timestamps
            target_duration: Target segment duration range (min, max)
            
        Returns:
            List of smart segments
        """
        if not words:
            return []
        
        segments = []
        current_segment = []
        current_start = words[0].start
        segment_id = 0
        
        min_duration, max_duration = target_duration
        
        for i, word in enumerate(words):
            current_segment.append(word)
            
            # Check for natural break points
            should_break = False
            
            # Duration-based break
            segment_duration = word.end - current_start
            if segment_duration >= max_duration:
                should_break = True
            
            # Pause-based break (>2 seconds)
            if i < len(words) - 1:
                pause_duration = words[i + 1].start - word.end
                if pause_duration > 2.0 and segment_duration >= min_duration:
                    should_break = True
            
            # Pattern-based break (after important patterns)
            if current_segment:
                segment_text = ' '.join(w.word for w in current_segment)
                patterns = self._detect_educational_patterns(segment_text, current_start, word.end)
                
                # Break after definitions or important points
                if patterns['patterns'].get('definitions') or patterns['patterns'].get('important_points'):
                    if segment_duration >= min_duration:
                        should_break = True
            
            if should_break and current_segment:
                # Create segment
                segment_text = ' '.join(w.word for w in current_segment)
                segment_end = current_segment[-1].end
                
                # Analyze segment for patterns
                patterns = self._detect_educational_patterns(segment_text, current_start, segment_end)
                
                # Determine segment type
                segment_type = 'content'
                if patterns['patterns'].get('definitions'):
                    segment_type = 'definition'
                elif patterns['patterns'].get('examples'):
                    segment_type = 'example'
                elif patterns['patterns'].get('important_points'):
                    segment_type = 'important'
                elif patterns['patterns'].get('transitions'):
                    segment_type = 'transition'
                elif patterns['patterns'].get('summaries'):
                    segment_type = 'summary'
                
                # Calculate confidence
                avg_confidence = sum(w.confidence for w in current_segment) / len(current_segment)
                
                segment = TranscriptSegment(
                    id=f"seg_{segment_id:04d}",
                    start=current_start,
                    end=segment_end,
                    text=segment_text,
                    segment_type=segment_type,
                    importance_score=patterns['importance_score'],
                    key_terms=patterns['key_terms'],
                    confidence=avg_confidence,
                    word_count=len(current_segment),
                    duration=segment_end - current_start
                )
                
                segments.append(segment)
                
                # Reset for next segment
                current_segment = []
                segment_id += 1
                if i < len(words) - 1:
                    current_start = words[i + 1].start
        
        # Handle remaining words
        if current_segment:
            segment_text = ' '.join(w.word for w in current_segment)
            segment_end = current_segment[-1].end
            
            patterns = self._detect_educational_patterns(segment_text, current_start, segment_end)
            
            segment_type = 'content'
            if patterns['patterns'].get('definitions'):
                segment_type = 'definition'
            elif patterns['patterns'].get('examples'):
                segment_type = 'example'
            elif patterns['patterns'].get('important_points'):
                segment_type = 'important'
            elif patterns['patterns'].get('transitions'):
                segment_type = 'transition'
            elif patterns['patterns'].get('summaries'):
                segment_type = 'summary'
            
            avg_confidence = sum(w.confidence for w in current_segment) / len(current_segment)
            
            segment = TranscriptSegment(
                id=f"seg_{segment_id:04d}",
                start=current_start,
                end=segment_end,
                text=segment_text,
                segment_type=segment_type,
                importance_score=patterns['importance_score'],
                key_terms=patterns['key_terms'],
                confidence=avg_confidence,
                word_count=len(current_segment),
                duration=segment_end - current_start
            )
            
            segments.append(segment)
        
        # Merge very short segments
        segments = self._merge_short_segments(segments, min_duration=10.0)
        
        return segments
    
    def _merge_short_segments(self, segments: List[TranscriptSegment], min_duration: float) -> List[TranscriptSegment]:
        """Merge segments that are too short."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            if current.duration < min_duration:
                # Merge with next segment
                current = TranscriptSegment(
                    id=current.id,
                    start=current.start,
                    end=next_segment.end,
                    text=f"{current.text} {next_segment.text}",
                    segment_type=current.segment_type if current.importance_score >= next_segment.importance_score else next_segment.segment_type,
                    importance_score=max(current.importance_score, next_segment.importance_score),
                    key_terms=list(set(current.key_terms + next_segment.key_terms)),
                    confidence=(current.confidence + next_segment.confidence) / 2,
                    word_count=current.word_count + next_segment.word_count,
                    duration=next_segment.end - current.start
                )
            else:
                merged.append(current)
                current = next_segment
        
        merged.append(current)
        return merged
    
    def _extract_key_learning_moments(self, segments: List[TranscriptSegment]) -> List[KeyLearningMoment]:
        """Extract key learning moments from segments."""
        key_moments = []
        
        for segment in segments:
            if segment.importance_score >= 0.7:  # High importance threshold
                moment_type = segment.segment_type
                if moment_type == 'content':
                    moment_type = 'important_point'
                
                # Find related segments (within 30 seconds)
                related_segments = []
                for other_segment in segments:
                    if (abs(other_segment.start - segment.start) <= 30.0 and 
                        other_segment.id != segment.id):
                        related_segments.append(other_segment.id)
                
                # Create context from surrounding segments
                context = ""
                for other_segment in segments:
                    if (other_segment.start >= segment.start - 60.0 and 
                        other_segment.end <= segment.end + 60.0 and
                        other_segment.id != segment.id):
                        context += f" {other_segment.text}"
                
                key_moment = KeyLearningMoment(
                    timestamp=segment.start,
                    type=moment_type,
                    text=segment.text,
                    importance_score=segment.importance_score,
                    related_segments=related_segments,
                    context=context.strip()
                )
                
                key_moments.append(key_moment)
        
        # Sort by importance score
        key_moments.sort(key=lambda x: x.importance_score, reverse=True)
        
        return key_moments[:20]  # Limit to top 20 moments
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray, chunk_start: float) -> List[WordSegment]:
        """Transcribe a single audio chunk."""
        try:
            # Pad or trim audio to 30 seconds
            audio_padded = pad_or_trim(audio_chunk)
            
            # Create mel spectrogram
            mel = log_mel_spectrogram(audio_padded)
            
            # Detect language (force English for educational content)
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            # Decode with word-level timestamps
            options = whisper.DecodingOptions(
                language="en",  # Force English for educational content
                fp16=False,
                without_timestamps=False
            )
            
            result = whisper.decode(self.model, mel, options)
            
            # Convert to word segments
            words = []
            if result.words:
                for word_info in result.words:
                    word = WordSegment(
                        word=word_info.word,
                        start=word_info.start + chunk_start,
                        end=word_info.end + chunk_start,
                        confidence=word_info.probability
                    )
                    words.append(word)
            
            return words
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return []
    
    def _chunk_audio(self, audio_path: str) -> List[Tuple[np.ndarray, float]]:
        """Split audio into chunks for processing."""
        try:
            audio = load_audio(audio_path)
            sample_rate = 16000  # Whisper sample rate
            
            chunk_samples = self.chunk_duration * sample_rate
            chunks = []
            
            for i in range(0, len(audio), int(chunk_samples)):
                chunk = audio[i:i + int(chunk_samples)]
                chunk_start = i / sample_rate
                chunks.append((chunk, chunk_start))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise AudioFileError(f"Failed to chunk audio: {e}")
    
    async def transcribe_audio(
        self, 
        audio_path: str, 
        progress_callback: Optional[Callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file with educational content understanding.
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback function
            
        Returns:
            TranscriptionResult with all metadata
            
        Raises:
            AudioFileError: If audio file is corrupted
            TranscriptionTimeoutError: If transcription times out
            MemoryError: If insufficient memory
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(audio_path)
            if self._is_cached(cache_key):
                cached_data = self._load_cache(cache_key)
                if cached_data:
                    logger.info(f"Using cached transcription for {audio_path}")
                    return TranscriptionResult(**cached_data)
            
            # Validate audio file
            if not os.path.exists(audio_path):
                raise AudioFileError(f"Audio file not found: {audio_path}")
            
            # Check file size for memory estimation
            file_size = os.path.getsize(audio_path)
            estimated_memory_gb = file_size / (1024 * 1024 * 1024) * 10  # Rough estimate
            
            if estimated_memory_gb > self.max_memory_gb:
                raise MemoryError(f"Estimated memory usage ({estimated_memory_gb:.1f}GB) exceeds limit ({self.max_memory_gb}GB)")
            
            logger.info(f"Starting transcription of {audio_path}")
            
            # Chunk audio for processing
            chunks = self._chunk_audio(audio_path)
            total_chunks = len(chunks)
            
            if progress_callback:
                progress_callback(0.0, "Starting transcription...")
            
            # Transcribe chunks in thread pool
            all_words = []
            completed_chunks = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunk transcription tasks
                future_to_chunk = {
                    executor.submit(self._transcribe_chunk, chunk, start): (chunk, start)
                    for chunk, start in chunks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_chunk, timeout=self.timeout_seconds):
                    chunk_words = future.result()
                    all_words.extend(chunk_words)
                    completed_chunks += 1
                    
                    if progress_callback:
                        progress = completed_chunks / total_chunks
                        progress_callback(progress, f"Transcribed {completed_chunks}/{total_chunks} chunks")
            
            if not all_words:
                raise TranscriptionError("No words transcribed from audio")
            
            # Sort words by timestamp
            all_words.sort(key=lambda w: w.start)
            
            # Create smart segments
            if progress_callback:
                progress_callback(0.8, "Creating smart segments...")
            
            segments = self._create_smart_segments(all_words)
            
            # Extract key learning moments
            if progress_callback:
                progress_callback(0.9, "Extracting key learning moments...")
            
            key_moments = self._extract_key_learning_moments(segments)
            
            # Prepare result
            full_text = ' '.join(w.word for w in all_words)
            total_duration = all_words[-1].end if all_words else 0.0
            average_confidence = sum(w.confidence for w in all_words) / len(all_words) if all_words else 0.0
            
            result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                key_moments=key_moments,
                language="en",  # Forced English for educational content
                language_confidence=1.0,
                total_duration=total_duration,
                word_count=len(all_words),
                average_confidence=average_confidence,
                processing_time=time.time() - start_time,
                cache_hit=False
            )
            
            # Save to cache
            self._save_cache(cache_key, asdict(result))
            
            if progress_callback:
                progress_callback(1.0, "Transcription completed")
            
            logger.info(f"Transcription completed in {result.processing_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            raise TranscriptionTimeoutError(f"Transcription timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Transcription failed: {e}")
    
    def get_transcription_status(self, audio_path: str) -> Dict:
        """Get transcription status for audio file."""
        try:
            cache_key = self._get_cache_key(audio_path)
            cache_path = self._get_cache_path(cache_key)
            
            return {
                'audio_path': audio_path,
                'is_cached': cache_path.exists(),
                'cache_size': cache_path.stat().st_size if cache_path.exists() else 0,
                'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get transcription status: {e}")
            return {}
    
    def cleanup_transcription(self, audio_path: str) -> bool:
        """Clean up cached transcription for audio file."""
        try:
            cache_key = self._get_cache_key(audio_path)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleaned up transcription cache for {audio_path}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup transcription: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get transcription cache statistics."""
        try:
            cache_files = list(self.cache_dir.rglob('*.json'))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_files_count': len(cache_files),
                'total_cache_size_bytes': total_size,
                'total_cache_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {} 