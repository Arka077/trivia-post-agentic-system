# 🧪 Tuesday Trivia Multi-Agent System

An intelligent multi-agent workflow system that discovers, curates, writes, and evaluates engaging scientific content for social media. Built with LangGraph, Mistral AI, and synchronized to Hugging Face Datasets for persistent cloud storage.

🚀 **[Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/arka7/Trivia-Agent)**

## ✨ Features

- **🤖 Multi-Agent Workflow**: Specialized agents for discovery, curation, writing, and quality control
- **☁️ Cloud Synchronization**: Automatic sync with Hugging Face Datasets
- **🔄 API Key Rotation**: Built-in fault tolerance with automatic key rotation
- **🎯 Quality Assurance**: Automated scoring and iterative refinement until posts meet quality standards
- **📊 Smart Deduplication**: Prevents duplicate topics using title similarity detection
- **🖥️ Interactive UI**: User-friendly Gradio interface with human-in-the-loop checkpoints

## 🏗️ Architecture

### Agent Pipeline

```
Discovery Agent (mistral-small)
    ↓
[Checkpoint 1: Review Candidates]
    ↓
Curator Agent (mistral-medium)
    ↓
[Checkpoint 2: Review Story Selection]
    ↓
Writer Agent (mistral-medium)
    ↓
Critic Agent (mistral-large) ←─┐
    ↓                            │
[Auto-Loop if score < 8.0] ─────┘
    ↓
[Checkpoint 3: Final Approval]
    ↓
Finalize & Save to Database + HF
```

### Agent Roles

1. **Discovery Agent** - Searches for recent scientific breakthroughs using web search tools
2. **Curator Agent** - Ranks candidates and selects the most engaging story
3. **Writer Agent** - Crafts compelling social media posts following strict formatting guidelines
4. **Critic Agent** - Evaluates quality (format, content, style) and automatically loops back for improvements

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Mistral AI API key(s)
- Tavily Search API key(s)
- Hugging Face account and token (optional, for cloud sync)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tuesday-trivia-agent.git
cd tuesday-trivia-agent

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Mistral AI API Keys (supports multiple for rotation)
MISTRAL_API_KEY=your_mistral_key_1
MISTRAL_API_KEY_2=your_mistral_key_2
MISTRAL_API_KEY_3=your_mistral_key_3

# Tavily Search API Keys
TAVILY_API_KEY=your_tavily_key_1
TAVILY_API_KEY_2=your_tavily_key_2

# Hugging Face Configuration (optional)
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_hf_username
HF_DATASET_REPO=tuesday-trivia-posts
```

### Running the Application

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

## 📋 Usage

1. **Start Discovery**: Enter a scientific topic (e.g., "quantum computing", "biotechnology")
2. **Review Candidates**: Approve the discovered breakthroughs or request a new search
3. **Review Story**: Approve the selected story or request a different one
4. **Review Draft**: The system automatically refines the post until quality score ≥ 8.0
5. **Final Approval**: Make final edits or approve for saving
6. **Copy & Share**: Copy the polished post to your clipboard

## 🛠️ Key Components

### Tools Available to Agents

- `search_science_breakthroughs` - Search scientific sources
- `fetch_article_content` - Retrieve full article text
- `get_all_previous_posts` - Access post history
- `check_topic_similarity` - Prevent duplicate content
- `get_example_posts_for_writer` - Reference formatting examples
- `save_approved_post` - Persist to database and HF

### Database Schema

```sql
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    source_url TEXT,
    quality_score REAL CHECK(quality_score >= 0 AND quality_score <= 10),
    engagement_score REAL,
    hashtags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
```

## 🎨 Post Format

The system generates posts following this structure:

```
[Compelling Title Following Examples]

[First paragraph: Context and background]
[Second paragraph: Technical details and impact]

#TuesdayTrivia #RnDCell #CCA #Topic1 #Topic2
```

**Specifications:**
- 140-180 words
- Technical yet accessible language
- Formal scientific tone
- Always includes core hashtags: `#TuesdayTrivia #RnDCell #CCA`

## ⚙️ Configuration

### Model Selection

The system uses tiered models for optimal performance:

```python
llm_small = "mistral-small-latest"    # Discovery
llm_medium = "mistral-medium-latest"  # Curation & Writing
llm_large = "mistral-large-latest"    # Quality Control
```

### Quality Thresholds

- **Auto-loop threshold**: Score < 8.0 triggers automatic rewrite
- **Max retries**: 3 attempts before requiring human intervention
- **Title similarity threshold**: 60% word overlap triggers duplicate warning

## 📊 Cloud Synchronization

Posts are automatically synchronized to Hugging Face Datasets:

- **Auto-sync**: After every save operation
- **Manual sync**: Available via UI button
- **Private repos**: Datasets are created as private by default
- **Bidirectional**: Syncs both to and from HF on startup

## 🔐 API Key Rotation

The system supports multiple API keys for fault tolerance:

```python
# Automatically rotates through available keys on failure
MISTRAL_API_KEY_1=key1
MISTRAL_API_KEY_2=key2
MISTRAL_API_KEY_3=key3
```

Benefits:
- Automatic failover on rate limits
- Load distribution across keys
- Improved reliability

## 🧪 Example Output

```markdown
Researchers Have Developed AI-Engineered Immune Cells to Target Cancer in Record Time

Researchers have developed an AI-driven approach that rapidly reprograms human immune cells to recognize and destroy cancer cells within weeks instead of months. By applying machine-learning models, the team designed highly specific T-cell receptors that accurately target cancer while minimizing damage to healthy tissue.

The AI system analyzed millions of possible receptor combinations to identify the most effective designs, enabling personalized immune therapies tailored to each patient's cancer type. This approach could significantly accelerate treatment development and move cancer therapy beyond one-size-fits-all solutions.

#TuesdayTrivia #CCA #RnDCell #CancerResearch #Immunotherapy #ArtificialIntelligence #PrecisionMedicine
```

## 📦 Dependencies

```
gradio>=4.0.0
langchain>=0.1.0
langchain-mistralai>=0.0.1
langchain-community>=0.0.1
langgraph>=0.0.1
huggingface-hub>=0.19.0
datasets>=2.15.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for powerful language models
- **Tavily** for scientific search capabilities
- **Hugging Face** for dataset hosting and collaboration tools
- **LangGraph** for agent orchestration framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the scientific community**
