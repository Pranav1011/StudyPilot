"""Initial database schema

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create videos table
    op.create_table('videos',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('youtube_id', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('channel', sa.String(length=200), nullable=False),
        sa.Column('channel_id', sa.String(length=50), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=False),
        sa.Column('upload_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=True),
        sa.Column('like_count', sa.Integer(), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('webpage_url', sa.String(length=500), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('is_educational', sa.Boolean(), nullable=False),
        sa.Column('educational_score', sa.Float(), nullable=False),
        sa.Column('detected_subjects', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('difficulty_level', sa.String(length=20), nullable=True),
        sa.Column('audio_path', sa.String(length=500), nullable=True),
        sa.Column('transcript_path', sa.String(length=500), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_videos_youtube_id', 'videos', ['youtube_id'], unique=False)
    op.create_index('idx_videos_status', 'videos', ['status'], unique=False)
    op.create_index('idx_videos_educational', 'videos', ['is_educational'], unique=False)
    op.create_index('idx_videos_upload_date', 'videos', ['upload_date'], unique=False)
    op.create_index('idx_videos_duration', 'videos', ['duration'], unique=False)
    
    # Create video_segments table
    op.create_table('video_segments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=False),
        sa.Column('segment_type', sa.String(length=20), nullable=False),
        sa.Column('importance_score', sa.Float(), nullable=False),
        sa.Column('key_terms', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_segments_video_time', 'video_segments', ['video_id', 'start_time'], unique=False)
    op.create_index('idx_segments_type', 'video_segments', ['segment_type'], unique=False)
    op.create_index('idx_segments_importance', 'video_segments', ['importance_score'], unique=False)
    op.create_index('idx_segments_embedding', 'video_segments', ['embedding'], unique=False, postgresql_using='gin')
    
    # Create concepts table
    op.create_table('concepts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('segment_id', sa.String(), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('definition', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('difficulty_level', sa.String(length=20), nullable=True),
        sa.Column('parent_concept_id', sa.String(), nullable=True),
        sa.Column('related_concept_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('importance_score', sa.Float(), nullable=False),
        sa.Column('frequency_count', sa.Integer(), nullable=False),
        sa.Column('first_mentioned_at', sa.Float(), nullable=False),
        sa.Column('last_mentioned_at', sa.Float(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['parent_concept_id'], ['concepts.id'], ),
        sa.ForeignKeyConstraint(['segment_id'], ['video_segments.id'], ),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('video_id', 'name', name='uq_concept_video_name')
    )
    op.create_index('idx_concepts_name', 'concepts', ['name'], unique=False)
    op.create_index('idx_concepts_video', 'concepts', ['video_id'], unique=False)
    op.create_index('idx_concepts_category', 'concepts', ['category'], unique=False)
    op.create_index('idx_concepts_importance', 'concepts', ['importance_score'], unique=False)
    op.create_index('idx_concepts_embedding', 'concepts', ['embedding'], unique=False, postgresql_using='gin')
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('first_name', sa.String(length=100), nullable=True),
        sa.Column('last_name', sa.String(length=100), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('preferred_subjects', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('difficulty_preference', sa.String(length=20), nullable=True),
        sa.Column('study_goals', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('notification_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('total_study_time', sa.Integer(), nullable=False),
        sa.Column('videos_watched', sa.Integer(), nullable=False),
        sa.Column('quizzes_completed', sa.Integer(), nullable=False),
        sa.Column('flashcards_reviewed', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_username', 'users', ['username'], unique=False)
    op.create_index('idx_users_active', 'users', ['is_active'], unique=False)
    
    # Create user_video_progress table
    op.create_table('user_video_progress',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('current_position', sa.Float(), nullable=False),
        sa.Column('total_watched', sa.Float(), nullable=False),
        sa.Column('watch_count', sa.Integer(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('comprehension_score', sa.Float(), nullable=True),
        sa.Column('retention_score', sa.Float(), nullable=True),
        sa.Column('engagement_score', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('bookmarks', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('last_studied_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'video_id', name='uq_user_video_progress')
    )
    op.create_index('idx_user_progress_user_video', 'user_video_progress', ['user_id', 'video_id'], unique=False)
    op.create_index('idx_user_progress_completed', 'user_video_progress', ['is_completed'], unique=False)
    op.create_index('idx_user_progress_last_studied', 'user_video_progress', ['last_studied_at'], unique=False)
    
    # Create user_concept_mastery table
    op.create_table('user_concept_mastery',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('concept_id', sa.String(), nullable=False),
        sa.Column('ease_factor', sa.Float(), nullable=False),
        sa.Column('interval', sa.Integer(), nullable=False),
        sa.Column('repetitions', sa.Integer(), nullable=False),
        sa.Column('quality', sa.Integer(), nullable=False),
        sa.Column('next_review_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('mastery_level', sa.Float(), nullable=False),
        sa.Column('is_mastered', sa.Boolean(), nullable=False),
        sa.Column('mastered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_reviews', sa.Integer(), nullable=False),
        sa.Column('correct_reviews', sa.Integer(), nullable=False),
        sa.Column('average_response_time', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['concept_id'], ['concepts.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'concept_id', name='uq_user_concept_mastery')
    )
    op.create_index('idx_concept_mastery_user_concept', 'user_concept_mastery', ['user_id', 'concept_id'], unique=False)
    op.create_index('idx_concept_mastery_next_review', 'user_concept_mastery', ['next_review_at'], unique=False)
    op.create_index('idx_concept_mastery_mastered', 'user_concept_mastery', ['is_mastered'], unique=False)
    
    # Create quizzes table
    op.create_table('quizzes',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('quiz_type', sa.String(length=20), nullable=False),
        sa.Column('difficulty_level', sa.String(length=20), nullable=True),
        sa.Column('total_questions', sa.Integer(), nullable=False),
        sa.Column('time_limit', sa.Integer(), nullable=True),
        sa.Column('passing_score', sa.Float(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('generation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_quizzes_video', 'quizzes', ['video_id'], unique=False)
    op.create_index('idx_quizzes_type', 'quizzes', ['quiz_type'], unique=False)
    op.create_index('idx_quizzes_difficulty', 'quizzes', ['difficulty_level'], unique=False)
    
    # Create quiz_questions table
    op.create_table('quiz_questions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('quiz_id', sa.String(), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('question_type', sa.String(length=20), nullable=False),
        sa.Column('correct_answer', sa.Text(), nullable=False),
        sa.Column('options', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('difficulty_level', sa.String(length=20), nullable=True),
        sa.Column('points', sa.Integer(), nullable=False),
        sa.Column('related_segment_id', sa.String(), nullable=True),
        sa.Column('related_concept_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('generation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['quiz_id'], ['quizzes.id'], ),
        sa.ForeignKeyConstraint(['related_segment_id'], ['video_segments.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_quiz_questions_quiz', 'quiz_questions', ['quiz_id'], unique=False)
    op.create_index('idx_quiz_questions_type', 'quiz_questions', ['question_type'], unique=False)
    op.create_index('idx_quiz_questions_difficulty', 'quiz_questions', ['difficulty_level'], unique=False)
    
    # Create flashcard_decks table
    op.create_table('flashcard_decks',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('total_cards', sa.Integer(), nullable=False),
        sa.Column('new_cards_per_day', sa.Integer(), nullable=False),
        sa.Column('review_cards_per_day', sa.Integer(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('generation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_flashcard_decks_video', 'flashcard_decks', ['video_id'], unique=False)
    op.create_index('idx_flashcard_decks_category', 'flashcard_decks', ['category'], unique=False)
    
    # Create flashcards table
    op.create_table('flashcards',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deck_id', sa.String(), nullable=False),
        sa.Column('front_text', sa.Text(), nullable=False),
        sa.Column('back_text', sa.Text(), nullable=False),
        sa.Column('hint', sa.Text(), nullable=True),
        sa.Column('ease_factor', sa.Float(), nullable=False),
        sa.Column('interval', sa.Integer(), nullable=False),
        sa.Column('repetitions', sa.Integer(), nullable=False),
        sa.Column('quality', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('next_review_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_reviews', sa.Integer(), nullable=False),
        sa.Column('correct_reviews', sa.Integer(), nullable=False),
        sa.Column('average_response_time', sa.Float(), nullable=True),
        sa.Column('related_segment_id', sa.String(), nullable=True),
        sa.Column('related_concept_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('generation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['deck_id'], ['flashcard_decks.id'], ),
        sa.ForeignKeyConstraint(['related_segment_id'], ['video_segments.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_flashcards_deck', 'flashcards', ['deck_id'], unique=False)
    op.create_index('idx_flashcards_status', 'flashcards', ['status'], unique=False)
    op.create_index('idx_flashcards_next_review', 'flashcards', ['next_review_at'], unique=False)
    op.create_index('idx_flashcards_due', 'flashcards', ['status', 'next_review_at'], unique=False)
    
    # Create study_guides table
    op.create_table('study_guides',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('difficulty_level', sa.String(length=20), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('key_points', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('related_concept_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('related_segment_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('generation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_study_guides_video', 'study_guides', ['video_id'], unique=False)
    op.create_index('idx_study_guides_difficulty', 'study_guides', ['difficulty_level'], unique=False)
    
    # Create learning_sessions table
    op.create_table('learning_sessions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('session_type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('content_ids', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('progress_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_completed', sa.Boolean(), nullable=False),
        sa.Column('completion_percentage', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_learning_sessions_user', 'learning_sessions', ['user_id'], unique=False)
    op.create_index('idx_learning_sessions_type', 'learning_sessions', ['session_type'], unique=False)
    op.create_index('idx_learning_sessions_started', 'learning_sessions', ['started_at'], unique=False)
    op.create_index('idx_learning_sessions_completed', 'learning_sessions', ['is_completed'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('learning_sessions')
    op.drop_table('study_guides')
    op.drop_table('flashcards')
    op.drop_table('flashcard_decks')
    op.drop_table('quiz_questions')
    op.drop_table('quizzes')
    op.drop_table('user_concept_mastery')
    op.drop_table('user_video_progress')
    op.drop_table('users')
    op.drop_table('concepts')
    op.drop_table('video_segments')
    op.drop_table('videos')
    
    # Drop pgvector extension
    op.execute('DROP EXTENSION IF EXISTS vector') 