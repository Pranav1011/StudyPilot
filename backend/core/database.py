"""
Database Models and Configuration

Comprehensive database schema for StudyPilot with SQLAlchemy async support,
PostgreSQL with pgvector extension, and Alembic migrations.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint, ARRAY,
    MetaData, create_engine, event
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY as PG_ARRAY
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, DisconnectionError, TimeoutError

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'echo': False,
    'echo_pool': False,
}

# Create base class for models
Base = declarative_base()

# Custom UUID type for PostgreSQL
class UUIDPrimaryKey:
    """Mixin for UUID primary keys."""
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4()),
        index=True
    )

# Timestamp mixin
class TimestampMixin:
    """Mixin for created/updated timestamps."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

# Soft delete mixin
class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False
    )

# Enums
class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SegmentType(str, Enum):
    DEFINITION = "definition"
    EXAMPLE = "example"
    IMPORTANT = "important"
    TRANSITION = "transition"
    SUMMARY = "summary"
    CONTENT = "content"

class QuizType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    MATCHING = "matching"
    ESSAY = "essay"

class FlashcardStatus(str, Enum):
    NEW = "new"
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"

# Database Models
class Video(Base, UUIDPrimaryKey, TimestampMixin):
    """YouTube video metadata and processing status."""
    __tablename__ = "videos"
    
    # YouTube metadata
    youtube_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    channel: Mapped[str] = mapped_column(String(200), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(50), nullable=False)
    duration: Mapped[int] = mapped_column(Integer, nullable=False)  # seconds
    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    view_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    webpage_url: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Processing status
    status: Mapped[VideoStatus] = mapped_column(String(20), default=VideoStatus.PENDING, nullable=False)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Educational analysis
    is_educational: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    educational_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    detected_subjects: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # File paths
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    transcript_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    segments = relationship("VideoSegment", back_populates="video", cascade="all, delete-orphan")
    concepts = relationship("Concept", back_populates="video", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="video", cascade="all, delete-orphan")
    flashcard_decks = relationship("FlashcardDeck", back_populates="video", cascade="all, delete-orphan")
    study_guides = relationship("StudyGuide", back_populates="video", cascade="all, delete-orphan")
    user_progress = relationship("UserVideoProgress", back_populates="video", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_videos_youtube_id', 'youtube_id'),
        Index('idx_videos_status', 'status'),
        Index('idx_videos_educational', 'is_educational'),
        Index('idx_videos_upload_date', 'upload_date'),
        Index('idx_videos_duration', 'duration'),
    )

class VideoSegment(Base, UUIDPrimaryKey, TimestampMixin):
    """Smart video segments with educational classifications."""
    __tablename__ = "video_segments"
    
    # Video relationship
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    
    # Segment metadata
    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    end_time: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    duration: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Educational classification
    segment_type: Mapped[SegmentType] = mapped_column(String(20), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    key_terms: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Embeddings for vector search
    embedding: Mapped[Optional[List[float]]] = mapped_column(PG_ARRAY(Float), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="segments")
    concepts = relationship("Concept", back_populates="segment", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_segments_video_time', 'video_id', 'start_time'),
        Index('idx_segments_type', 'segment_type'),
        Index('idx_segments_importance', 'importance_score'),
        Index('idx_segments_embedding', 'embedding', postgresql_using='gin'),
    )

class Concept(Base, UUIDPrimaryKey, TimestampMixin):
    """Extracted concepts with relationships."""
    __tablename__ = "concepts"
    
    # Video and segment relationships
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    segment_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("video_segments.id"), nullable=True)
    
    # Concept metadata
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    definition: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Concept relationships
    parent_concept_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("concepts.id"), nullable=True)
    related_concept_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)
    
    # Educational metadata
    importance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    frequency_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    first_mentioned_at: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    last_mentioned_at: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    
    # Embeddings for vector search
    embedding: Mapped[Optional[List[float]]] = mapped_column(PG_ARRAY(Float), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="concepts")
    segment = relationship("VideoSegment", back_populates="concepts")
    parent_concept = relationship("Concept", remote_side=[id])
    child_concepts = relationship("Concept", back_populates="parent_concept")
    user_mastery = relationship("UserConceptMastery", back_populates="concept", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_concepts_name', 'name'),
        Index('idx_concepts_video', 'video_id'),
        Index('idx_concepts_category', 'category'),
        Index('idx_concepts_importance', 'importance_score'),
        Index('idx_concepts_embedding', 'embedding', postgresql_using='gin'),
        UniqueConstraint('video_id', 'name', name='uq_concept_video_name'),
    )

class User(Base, UUIDPrimaryKey, TimestampMixin, SoftDeleteMixin):
    """User authentication and preferences."""
    __tablename__ = "users"
    
    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Profile
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Preferences
    preferred_subjects: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    difficulty_preference: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    study_goals: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    notification_settings: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Learning statistics
    total_study_time: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # seconds
    videos_watched: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quizzes_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    flashcards_reviewed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Relationships
    video_progress = relationship("UserVideoProgress", back_populates="user", cascade="all, delete-orphan")
    concept_mastery = relationship("UserConceptMastery", back_populates="user", cascade="all, delete-orphan")
    learning_sessions = relationship("LearningSession", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_username', 'username'),
        Index('idx_users_active', 'is_active'),
    )

class UserVideoProgress(Base, UUIDPrimaryKey, TimestampMixin):
    """Detailed user progress tracking for videos."""
    __tablename__ = "user_video_progress"
    
    # Relationships
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    
    # Progress tracking
    current_position: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # seconds
    total_watched: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # seconds
    watch_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Learning metrics
    comprehension_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retention_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Study data
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bookmarks: Mapped[List[float]] = mapped_column(PG_ARRAY(Float), nullable=True)  # timestamps
    last_studied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="video_progress")
    video = relationship("Video", back_populates="user_progress")
    
    __table_args__ = (
        Index('idx_user_progress_user_video', 'user_id', 'video_id'),
        Index('idx_user_progress_completed', 'is_completed'),
        Index('idx_user_progress_last_studied', 'last_studied_at'),
        UniqueConstraint('user_id', 'video_id', name='uq_user_video_progress'),
    )

class UserConceptMastery(Base, UUIDPrimaryKey, TimestampMixin):
    """Concept-level mastery tracking for users."""
    __tablename__ = "user_concept_mastery"
    
    # Relationships
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    concept_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("concepts.id"), nullable=False)
    
    # Mastery tracking (SM-2 algorithm)
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5, nullable=False)
    interval: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # days
    repetitions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quality: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0-5 scale
    
    # Next review
    next_review_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Mastery level
    mastery_level: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0.0 to 1.0
    is_mastered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mastered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Study history
    total_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_response_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    
    # Relationships
    user = relationship("User", back_populates="concept_mastery")
    concept = relationship("Concept", back_populates="user_mastery")
    
    __table_args__ = (
        Index('idx_concept_mastery_user_concept', 'user_id', 'concept_id'),
        Index('idx_concept_mastery_next_review', 'next_review_at'),
        Index('idx_concept_mastery_mastered', 'is_mastered'),
        UniqueConstraint('user_id', 'concept_id', name='uq_user_concept_mastery'),
    )

class Quiz(Base, UUIDPrimaryKey, TimestampMixin):
    """Generated quizzes for videos."""
    __tablename__ = "quizzes"
    
    # Video relationship
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    
    # Quiz metadata
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quiz_type: Mapped[QuizType] = mapped_column(String(20), nullable=False)
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Quiz configuration
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    time_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # minutes
    passing_score: Mapped[float] = mapped_column(Float, default=70.0, nullable=False)  # percentage
    
    # Generation metadata
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="quizzes")
    questions = relationship("QuizQuestion", back_populates="quiz", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_quizzes_video', 'video_id'),
        Index('idx_quizzes_type', 'quiz_type'),
        Index('idx_quizzes_difficulty', 'difficulty_level'),
    )

class QuizQuestion(Base, UUIDPrimaryKey, TimestampMixin):
    """Individual quiz questions."""
    __tablename__ = "quiz_questions"
    
    # Quiz relationship
    quiz_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("quizzes.id"), nullable=False)
    
    # Question metadata
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[QuizType] = mapped_column(String(20), nullable=False)
    correct_answer: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)  # for multiple choice
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Difficulty and scoring
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    points: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Related content
    related_segment_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("video_segments.id"), nullable=True)
    related_concept_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)
    
    # Generation metadata
    generation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    quiz = relationship("Quiz", back_populates="questions")
    
    __table_args__ = (
        Index('idx_quiz_questions_quiz', 'quiz_id'),
        Index('idx_quiz_questions_type', 'question_type'),
        Index('idx_quiz_questions_difficulty', 'difficulty_level'),
    )

class FlashcardDeck(Base, UUIDPrimaryKey, TimestampMixin):
    """Flashcard decks for spaced repetition."""
    __tablename__ = "flashcard_decks"
    
    # Video relationship
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    
    # Deck metadata
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Deck configuration
    total_cards: Mapped[int] = mapped_column(Integer, nullable=False)
    new_cards_per_day: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    review_cards_per_day: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    
    # Generation metadata
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="flashcard_decks")
    flashcards = relationship("Flashcard", back_populates="deck", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_flashcard_decks_video', 'video_id'),
        Index('idx_flashcard_decks_category', 'category'),
    )

class Flashcard(Base, UUIDPrimaryKey, TimestampMixin):
    """Individual flashcards for spaced repetition."""
    __tablename__ = "flashcards"
    
    # Deck relationship
    deck_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("flashcard_decks.id"), nullable=False)
    
    # Card content
    front_text: Mapped[str] = mapped_column(Text, nullable=False)
    back_text: Mapped[str] = mapped_column(Text, nullable=False)
    hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Spaced repetition data (SM-2 algorithm)
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5, nullable=False)
    interval: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # days
    repetitions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quality: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0-5 scale
    
    # Review scheduling
    status: Mapped[FlashcardStatus] = mapped_column(String(20), default=FlashcardStatus.NEW, nullable=False)
    next_review_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Review history
    total_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_response_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    
    # Related content
    related_segment_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey("video_segments.id"), nullable=True)
    related_concept_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)
    
    # Generation metadata
    generation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    deck = relationship("FlashcardDeck", back_populates="flashcards")
    
    __table_args__ = (
        Index('idx_flashcards_deck', 'deck_id'),
        Index('idx_flashcards_status', 'status'),
        Index('idx_flashcards_next_review', 'next_review_at'),
        Index('idx_flashcards_due', 'status', 'next_review_at'),
    )

class StudyGuide(Base, UUIDPrimaryKey, TimestampMixin):
    """Comprehensive study guides for videos."""
    __tablename__ = "study_guides"
    
    # Video relationship
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False)
    
    # Guide metadata
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Guide content
    content: Mapped[str] = mapped_column(Text, nullable=False)  # Markdown content
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_points: Mapped[List[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    
    # Related content
    related_concept_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)
    related_segment_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)
    
    # Generation metadata
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="study_guides")
    
    __table_args__ = (
        Index('idx_study_guides_video', 'video_id'),
        Index('idx_study_guides_difficulty', 'difficulty_level'),
    )

class LearningSession(Base, UUIDPrimaryKey, TimestampMixin):
    """Study session data for analytics."""
    __tablename__ = "learning_sessions"
    
    # User relationship
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'video_watch', 'quiz', 'flashcard', 'review'
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Session timing
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # seconds
    
    # Session data
    content_ids: Mapped[List[str]] = mapped_column(PG_ARRAY(UUID(as_uuid=False)), nullable=True)  # related video/quiz/etc IDs
    progress_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    performance_metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Session state
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completion_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="learning_sessions")
    
    __table_args__ = (
        Index('idx_learning_sessions_user', 'user_id'),
        Index('idx_learning_sessions_type', 'session_type'),
        Index('idx_learning_sessions_started', 'started_at'),
        Index('idx_learning_sessions_completed', 'is_completed'),
    )


# Database Connection and Session Management
class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize database connection and session factory."""
        async with self._lock:
            if self.engine is not None:
                return
            
            try:
                logger.info("Initializing database connection...")
                
                # Create async engine with connection pool
                self.engine = create_async_engine(
                    self.database_url,
                    **DATABASE_CONFIG,
                    future=True
                )
                
                # Create session factory
                self.session_factory = async_sessionmaker(
                    bind=self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False
                )
                
                # Test connection
                async with self.engine.begin() as conn:
                    await conn.execute("SELECT 1")
                
                logger.info("Database connection initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
    
    async def close(self):
        """Close database connections."""
        async with self._lock:
            if self.engine:
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None
                logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            await self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connection status."""
        try:
            if not self.engine:
                return {"status": "not_initialized", "error": "Database not initialized"}
            
            async with self.engine.begin() as conn:
                # Check basic connectivity
                result = await conn.execute("SELECT 1 as test")
                test_result = result.scalar()
                
                if test_result != 1:
                    return {"status": "error", "error": "Basic query failed"}
                
                # Check pgvector extension
                result = await conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                vector_extension = result.scalar()
                
                # Get connection pool stats
                pool = self.engine.pool
                pool_stats = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
                
                return {
                    "status": "healthy",
                    "vector_extension": bool(vector_extension),
                    "pool_stats": pool_stats,
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else "unknown"
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_tables(self):
        """Create all database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise


# Database utilities and helpers
class DatabaseUtils:
    """Database utility functions."""
    
    @staticmethod
    async def retry_on_deadlock(func, max_retries: int = 3, delay: float = 0.1):
        """Retry function on deadlock with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if "deadlock" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Deadlock detected, retrying in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                raise
    
    @staticmethod
    async def with_timeout(func, timeout_seconds: float = 30.0):
        """Execute function with timeout."""
        try:
            return await asyncio.wait_for(func, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Database operation timed out after {timeout_seconds}s")
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID string."""
        return str(uuid.uuid4())
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)


# Seed data functions
class DatabaseSeeder:
    """Database seeding utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def seed_test_data(self):
        """Seed database with test data."""
        try:
            async with self.db_manager.get_session() as session:
                # Create test user
                test_user = User(
                    email="test@studypilot.com",
                    username="testuser",
                    hashed_password="hashed_password_here",
                    first_name="Test",
                    last_name="User",
                    is_verified=True
                )
                session.add(test_user)
                await session.flush()
                
                # Create test video
                test_video = Video(
                    youtube_id="dQw4w9WgXcQ",
                    title="Test Educational Video",
                    channel="Test Channel",
                    channel_id="UC123456789",
                    duration=600,
                    upload_date=datetime.now(timezone.utc),
                    webpage_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
                    is_educational=True,
                    educational_score=0.8,
                    status=VideoStatus.COMPLETED
                )
                session.add(test_video)
                await session.flush()
                
                # Create test segment
                test_segment = VideoSegment(
                    video_id=test_video.id,
                    start_time=0.0,
                    end_time=30.0,
                    duration=30.0,
                    text="This is a test segment for educational content.",
                    word_count=8,
                    segment_type=SegmentType.CONTENT,
                    importance_score=0.5,
                    confidence=0.9
                )
                session.add(test_segment)
                
                await session.commit()
                logger.info("Test data seeded successfully")
                
        except Exception as e:
            logger.error(f"Failed to seed test data: {e}")
            raise
    
    async def clear_test_data(self):
        """Clear test data from database."""
        try:
            async with self.db_manager.get_session() as session:
                # Delete test data in reverse dependency order
                await session.execute("DELETE FROM learning_sessions WHERE user_id IN (SELECT id FROM users WHERE email = 'test@studypilot.com')")
                await session.execute("DELETE FROM user_concept_mastery WHERE user_id IN (SELECT id FROM users WHERE email = 'test@studypilot.com')")
                await session.execute("DELETE FROM user_video_progress WHERE user_id IN (SELECT id FROM users WHERE email = 'test@studypilot.com')")
                await session.execute("DELETE FROM users WHERE email = 'test@studypilot.com'")
                
                await session.execute("DELETE FROM video_segments WHERE video_id IN (SELECT id FROM videos WHERE youtube_id = 'dQw4w9WgXcQ')")
                await session.execute("DELETE FROM videos WHERE youtube_id = 'dQw4w9WgXcQ'")
                
                await session.commit()
                logger.info("Test data cleared successfully")
                
        except Exception as e:
            logger.error(f"Failed to clear test data: {e}")
            raise


# Backup and restore utilities
class DatabaseBackup:
    """Database backup and restore utilities."""
    
    def __init__(self, db_manager: DatabaseManager, backup_dir: str = "./backups"):
        self.db_manager = db_manager
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create database backup."""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        backup_path = self.backup_dir / backup_name
        
        try:
            # Extract database connection info
            db_url = self.db_manager.database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            
            # Create backup using pg_dump
            import subprocess
            
            cmd = [
                "pg_dump",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges",
                "--format=plain",
                f"--file={backup_path}",
                db_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Backup failed: {result.stderr}")
            
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    async def restore_backup(self, backup_path: str):
        """Restore database from backup."""
        try:
            # Extract database connection info
            db_url = self.db_manager.database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            
            # Restore using psql
            import subprocess
            
            cmd = [
                "psql",
                "--quiet",
                "--file", backup_path,
                db_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")
            
            logger.info(f"Database restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database() first.")
    return _db_manager

async def initialize_database(database_url: str):
    """Initialize global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    await _db_manager.initialize()

async def close_database():
    """Close global database manager."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None 