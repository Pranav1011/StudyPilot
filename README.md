# StudyPilot 🚀

An AI-powered study companion that transforms YouTube educational videos into a complete interactive learning experience with quizzes, flashcards, personalized tutoring, and progress tracking.

## 🎯 Core Innovation

Unlike simple transcription tools, StudyPilot understands what's being **TAUGHT**, not just what's being said, and automatically generates comprehensive study materials.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+, PostgreSQL, Redis, ChromaDB
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, React Query
- **AI/ML**: Whisper (local), Ollama (local LLM), Sentence Transformers
- **Infrastructure**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js 18+
- Git

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StudyPilot
   ```

2. **Copy environment template**
   ```bash
   cp env.example .env
   cp frontend/env.local frontend/.env.local
   ```

3. **Configure environment variables**
   Edit `.env` and `frontend/.env.local` with your configuration.

### Docker Setup

1. **Build and start all services**
   ```bash
   docker-compose up --build
   ```

2. **For development with hot reload**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Manual Setup (Development)

1. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Database Setup**
   ```bash
   python scripts/setup_db.py
   ```

## 📁 Project Structure

```
studypilot/
├── backend/                 # FastAPI backend
│   ├── core/               # Core video processing
│   ├── agents/             # AI agents for content generation
│   ├── learning/           # Learning algorithms & tracking
│   ├── services/           # External services integration
│   ├── api/                # API routes & endpoints
│   ├── schemas/            # Pydantic models
│   └── utils/              # Utility functions
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # App router pages
│   │   ├── components/    # React components
│   │   ├── lib/           # Utilities & hooks
│   │   └── styles/        # Global styles
├── mcp/                   # Model Context Protocol server
├── scripts/               # Utility scripts
├── docker/                # Docker configurations
└── .github/               # CI/CD workflows
```

## 🔧 Development Workflow

### Backend Development

1. **Run tests**
   ```bash
   cd backend
   pytest
   ```

2. **Code formatting**
   ```bash
   black .
   isort .
   ```

3. **Linting**
   ```bash
   flake8 .
   mypy .
   ```

### Frontend Development

1. **Run tests**
   ```bash
   cd frontend
   npm test
   ```

2. **Linting**
   ```bash
   npm run lint
   ```

3. **Type checking**
   ```bash
   npm run type-check
   ```

## 📚 API Documentation

### Core Endpoints

- `POST /api/videos/process` - Process YouTube video
- `GET /api/videos/{id}` - Get video details
- `POST /api/query` - Ask questions about video content
- `GET /api/learning/quiz/{video_id}` - Get generated quiz
- `GET /api/learning/flashcards/{video_id}` - Get flashcards
- `GET /api/progress/{user_id}` - Get learning progress

### Authentication

StudyPilot uses JWT tokens for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Integration Tests
```bash
docker-compose -f docker-compose.test.yml up --build
```

## 🚀 Deployment

### Production Deployment

1. **Build production images**
   ```bash
   docker-compose -f docker-compose.prod.yml build
   ```

2. **Deploy with environment variables**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Environment Variables

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/studypilot
REDIS_URL=redis://localhost:6379

# AI Services
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_DB_PATH=./chroma_db

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# External Services
YOUTUBE_API_KEY=your-youtube-api-key
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   npm run test
   pytest
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **TypeScript**: Use ESLint and Prettier
- **Commits**: Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.studypilot.ai](https://docs.studypilot.ai)
- **Issues**: [GitHub Issues](https://github.com/studypilot/studypilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/studypilot/studypilot/discussions)

## 🙏 Acknowledgments

- OpenAI Whisper for transcription
- Ollama for local LLM inference
- ChromaDB for vector storage
- The open-source community for inspiration

---