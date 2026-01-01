import os
import json
import sqlite3
import random
import requests
import asyncio
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import tempfile
import shutil

# Third-party imports
import gradio as gr
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from huggingface_hub import HfApi, hf_hub_download, upload_file
from datasets import Dataset, load_dataset

# --- 1. SETUP & CONFIGURATION ---

load_dotenv()

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "tuesday-trivia-posts")
HF_USERNAME = os.getenv("HF_USERNAME", None)

# Auto-construct repo name if not provided
if "/" not in HF_DATASET_REPO and HF_USERNAME:
    HF_DATASET_REPO = f"{HF_USERNAME}/{HF_DATASET_REPO}"

# Local temporary database
DB_DIR = Path(tempfile.gettempdir()) / "trivia_data"
DB_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = DB_DIR / "trivia_posts.db"

print(f"📁 Local database location: {DB_PATH.absolute()}")
print(f"☁️  HF Dataset repository: {HF_DATASET_REPO}")

# Initialize HF API
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# --- 2. API KEY ROTATION ---

class APIKeyRotator:
    """Handles API key rotation for fault tolerance"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.keys = self._load_keys()
        self.current_index = 0
        
    def _load_keys(self) -> List[str]:
        keys = []
        i = 1
        while True:
            key = os.getenv(f"{self.service_name}_API_KEY_{i}")
            if not key:
                if i == 1:
                    key = os.getenv(f"{self.service_name}_API_KEY")
                    if key:
                        keys.append(key)
                break
            keys.append(key)
            i += 1
        
        if not keys:
            print(f"⚠️  No API keys found for {self.service_name}. Ensure secrets are set.")
            return [] 
        
        print(f"✅ Loaded {len(keys)} API key(s) for {self.service_name}")
        random.shuffle(keys)
        return keys
    
    def get_key(self) -> str:
        if not self.keys: return ""
        return self.keys[self.current_index]
    
    def rotate(self) -> str:
        if not self.keys: return ""
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"🔄 Rotated {self.service_name} API key to index {self.current_index + 1}/{len(self.keys)}")
        return self.get_key()

mistral_rotator = APIKeyRotator("MISTRAL")
tavily_rotator = APIKeyRotator("TAVILY")

os.environ["MISTRAL_API_KEY"] = mistral_rotator.get_key()
os.environ["TAVILY_API_KEY"] = tavily_rotator.get_key()

# --- 3. RESILIENT LLM ---

def create_llm_with_rotation(model: str, temperature: float = 0.2, max_retries: int = 3):
    class ResilientLLM:
        def __init__(self, model, temperature):
            self.model = model
            self.temperature = temperature
            
        def invoke(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    llm = ChatMistralAI(
                        model=self.model,
                        temperature=self.temperature,
                        mistral_api_key=mistral_rotator.get_key()
                    )
                    return llm.invoke(*args, **kwargs)
                except Exception as e:
                    print(f"❌ API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        mistral_rotator.rotate()
                    else:
                        raise Exception(f"All API keys exhausted after {max_retries} attempts")
        
        def bind_tools(self, tools):
            class ToolBoundLLM:
                def __init__(self, parent, tools):
                    self.parent = parent
                    self.tools = tools
                
                def invoke(self, *args, **kwargs):
                    for attempt in range(max_retries):
                        try:
                            llm = ChatMistralAI(
                                model=self.parent.model,
                                temperature=self.parent.temperature,
                                mistral_api_key=mistral_rotator.get_key()
                            )
                            bound_llm = llm.bind_tools(self.tools)
                            return bound_llm.invoke(*args, **kwargs)
                        except Exception as e:
                            print(f"❌ API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                            if attempt < max_retries - 1:
                                mistral_rotator.rotate()
                            else:
                                raise Exception(f"All API keys exhausted after {max_retries} attempts")
            return ToolBoundLLM(self, tools)
    
    return ResilientLLM(model, temperature)

llm_small = create_llm_with_rotation("mistral-small-latest", temperature=0.2)
llm_medium = create_llm_with_rotation("mistral-medium-latest", temperature=0.2)
llm_large = create_llm_with_rotation("mistral-large-latest", temperature=0.2)

# --- 4. DATABASE & HF SYNC (keeping original code) ---

def get_db_connection():
    return sqlite3.connect(str(DB_PATH))

def sync_from_hf():
    if not HF_TOKEN or not hf_api:
        print("⚠️  No HF_TOKEN found, skipping sync from HF")
        return False
    try:
        print("📥 Syncing database from Hugging Face...")
        try:
            dataset = load_dataset(HF_DATASET_REPO, split="train", token=HF_TOKEN)
            if len(dataset) == 0:
                print("ℹ️  Dataset exists but is empty")
                return False
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM posts')
            for row in dataset:
                cursor.execute('''INSERT INTO posts (id, date, topic, summary, source_url, quality_score, engagement_score, hashtags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (row.get('id'), row.get('date'), row.get('topic'), row.get('summary'), row.get('source_url'), row.get('quality_score'), row.get('engagement_score'), row.get('hashtags'), row.get('created_at'), row.get('updated_at')))
            conn.commit()
            conn.close()
            print(f"✅ Synced {len(dataset)} posts from HF Dataset")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "doesn't exist" in error_msg or "doesn't contain any data files" in error_msg:
                print(f"ℹ️  Dataset '{HF_DATASET_REPO}' doesn't exist yet or is empty, will create on first save")
                return False
            else:
                print(f"⚠️  Error loading dataset: {e}")
                return False
    except Exception as e:
        print(f"⚠️  Error syncing from HF: {e}")
        return False

def sync_to_hf():
    if not HF_TOKEN or not hf_api:
        print("⚠️  No HF_TOKEN found, skipping sync to HF")
        return False
    try:
        print("📤 Syncing database to Hugging Face...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM posts')
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            print("ℹ️  No posts to sync")
            return False
        data = {col: [] for col in columns}
        for row in rows:
            for col, value in zip(columns, row):
                data[col].append(value)
        dataset = Dataset.from_dict(data)
        try:
            dataset.push_to_hub(HF_DATASET_REPO, token=HF_TOKEN, private=True)
            print(f"✅ Synced {len(rows)} posts to HF Dataset: {HF_DATASET_REPO}")
            return True
        except Exception as push_error:
            if "not found" in str(push_error).lower():
                print(f"📦 Creating new dataset repository: {HF_DATASET_REPO}")
                hf_api.create_repo(repo_id=HF_DATASET_REPO, token=HF_TOKEN, repo_type="dataset", private=True)
                dataset.push_to_hub(HF_DATASET_REPO, token=HF_TOKEN, private=True)
                print(f"✅ Created and synced to new dataset: {HF_DATASET_REPO}")
                return True
            else:
                raise push_error
    except Exception as e:
        print(f"❌ Error syncing to HF: {e}")
        return False

def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, topic TEXT NOT NULL, summary TEXT NOT NULL, source_url TEXT, quality_score REAL CHECK(quality_score >= 0 AND quality_score <= 10), engagement_score REAL, hashtags TEXT, created_at TEXT NOT NULL, updated_at TEXT)''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON posts(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality_score ON posts(quality_score)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON posts(created_at)')
    cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT NOT NULL)''')
    cursor.execute('''INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES ('db_version', '1.0', ?)''', (datetime.now().isoformat(),))
    conn.commit()
    conn.close()
    sync_from_hf()

init_database()

# --- 5. TOOLS (keeping original except check_topic_similarity) ---

@tool
def search_science_breakthroughs(query: str) -> str:
    """Search for recent scientific breakthroughs."""
    try:
        search = TavilySearchResults(max_results=10, include_domains=["sciencedaily.com", "nature.com", "science.org","technologyreview.com"], search_depth="advanced")
        results = search.invoke(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error in search: {str(e)}"

@tool
def fetch_article_content(url: str) -> str:
    """Fetch the full text content of an article."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:5000] if len(text) > 5000 else text
    except Exception as e:
        return f"Error fetching article: {str(e)}"

@tool
def get_all_previous_posts() -> str:
    """Retrieve all previously published posts with their titles."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, date, topic, summary, source_url, quality_score FROM posts ORDER BY date DESC')
        posts = cursor.fetchall()
        if not posts: return "No previous posts found."
        formatted = []
        for p in posts:
            # Extract title (first line of summary)
            title = p[3].split('\n')[0].strip() if p[3] else p[2]
            formatted.append({"id": p[0], "date": p[1], "title": title, "topic": p[2], "summary_preview": p[3][:100], "source_url": p[4], "quality_score": p[5]})
        return json.dumps(formatted, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn: conn.close()

@tool
def get_top_quality_posts(limit: int = 5) -> str:
    """Retrieve top quality posts."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, date, topic, summary, quality_score, hashtags FROM posts WHERE quality_score >= 7 ORDER BY quality_score DESC LIMIT ?', (limit,))
        posts = cursor.fetchall()
        if not posts: return "No high-quality example posts found."
        formatted = []
        for p in posts:
            formatted.append({"id": p[0], "date": p[1], "topic": p[2], "summary": p[3], "quality_score": p[4], "hashtags": p[5]})
        return json.dumps(formatted, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn: conn.close()

@tool
def save_approved_post(topic: str, summary: str, source_url: str, quality_score: float, hashtags: str) -> str:
    """Save an approved post to the database and sync to HF."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now()
        cursor.execute('''INSERT INTO posts (date, topic, summary, source_url, quality_score, hashtags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (now.strftime('%Y-%m-%d'), topic, summary, source_url, quality_score, hashtags, now.isoformat(), now.isoformat()))
        conn.commit()
        post_id = cursor.lastrowid
        conn.close()
        sync_result = sync_to_hf()
        sync_msg = "and synced to HF Dataset ☁️" if sync_result else "(HF sync skipped)"
        return f"Post saved successfully with ID: {post_id} {sync_msg}"
    except Exception as e:
        return f"Error saving post: {str(e)}"
    finally:
        if conn: 
            try:
                conn.close()
            except:
                pass

@tool
def check_topic_similarity(new_topic: str) -> str:
    """CRITICAL: Check if the TITLE is too similar to previous post titles. Focus on TITLE similarity only."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT topic, summary FROM posts')
        previous = cursor.fetchall()
        if not previous: return json.dumps({"is_duplicate": False, "similar_posts": [], "checked_title": new_topic})
        
        # Extract title from new topic (first line before any newlines)
        new_title = new_topic.split('\n')[0].strip().lower()
        new_keywords = set(word for word in new_title.split() if len(word) > 3)  # Ignore small words
        
        similar = []
        for prev_topic, prev_summary in previous:
            # Extract title from previous post (first line)
            prev_title = prev_summary.split('\n')[0].strip().lower()
            prev_keywords = set(word for word in prev_title.split() if len(word) > 3)
            
            if not new_keywords: continue
            overlap = len(new_keywords & prev_keywords)
            similarity = overlap / len(new_keywords) if new_keywords else 0
            
            # Higher threshold for title similarity (0.6 = 60% word overlap)
            if similarity > 0.6:
                similar.append({
                    "previous_title": prev_title,
                    "similarity_percentage": round(similarity * 100, 1),
                    "matching_words": list(new_keywords & prev_keywords)
                })
        
        return json.dumps({
            "is_duplicate": len(similar) > 0, 
            "similar_posts": similar,
            "new_title_checked": new_title,
            "warning": "⚠️ TITLE is too similar to existing posts!" if similar else "✅ Title is unique"
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn: conn.close()

@tool
def count_words(text: str) -> str:
    """Count the number of words in the given text."""
    return f"Word count: {len(text.split())}"

@tool
def get_example_posts_for_writer() -> str:
    """Get example posts to guide the writer agent."""
    examples = [
    {
        "title": "Scientists Found 14 Mysterious Creatures in the Ocean's Darkest Depths",
        "content": """Scientists Found 14 Mysterious Creatures in the Ocean's Darkest Depths

Scientists have just revealed 14 new species living miles beneath the ocean's surface—some found at depths over 6,000 meters! This discovery is part of the Ocean Species Discoveries initiative, which is changing how new marine life is identified and shared with the world. Among the surprises: a record-setting deep-sea mollusk, a carnivorous bivalve, and even a popcorn-shaped parasitic isopod. Each of these creatures shows just how much life still hides in the ocean's darkest corners.
What's even cooler? The team is using cutting-edge lab techniques to make classifying species faster, more open, and globally collaborative—helping scientists everywhere explore and protect our planet's last great frontier: the deep sea.

#TuesdayTrivia #RnDCell #CCA #OceanDiscovery #MarineBiology #DeepSea"""
    },
    {
        "title": "Researchers Have Developed Next-Generation Plant Immune System to Combat Bacterial Diseases",
        "content": """Researchers Have Developed Next-Generation Plant Immune System to Combat Bacterial Diseases

Researchers at the University of California, Davis have used artificial intelligence to boost plant immunity, helping crops like tomatoes and potatoes resist bacterial infections. Using AlphaFold, an AI system that predicts protein structures, they redesigned the immune receptor FLS2, which identifies flagellin, a bacterial movement protein. Since bacteria mutate flagellin to escape detection, the AI-guided redesign enabled plants to recognize more bacterial variants and strengthen their defenses.
By altering key amino acids, the team reactivated weakened receptors, restoring the plants’ ability to detect pathogens. This innovation could provide broad-spectrum disease resistance in major crops, including protection against Ralstonia solanacearum, the bacterium that causes bacterial wilt. The researchers now plan to extend this strategy to other plants using machine learning.

#CCA_RnD #TuesdayTrivia #ArtificialIntelligence #ProteinEngineering #PlantImmunity #AgriTech #BacterialWilt #GreenTech"""
    },
    {
        "title": "Researchers Have Developed a Flexible, Low-Cost Robotic Skin That Allows Machines to Sense Touch Like Humans",
        "content": """Researchers Have Developed a Flexible, Low-Cost Robotic Skin That Allows Machines to Sense Touch Like Humans

Scientists at the University of Cambridge have created a revolutionary skin-like material that brings us a step closer to giving robots a true sense of touch. This soft, flexible “robotic skin” is made from a smart gel that covers the entire surface of a robot, acting as a single, large sensor. Unlike older technologies that required hundreds of tiny sensors, this material can detect pressure, heat, and even pain—all at the same time and across multiple areas.
This breakthrough means robots can now respond more naturally and sensitively to their environment. They can handle delicate objects with care or react safely to human contact. This is especially important in settings like hospitals, elderly care, or even homes, where robots may assist people directly. Just like human skin helps us feel and respond to the world around us, this innovation allows machines to become more aware, making them more helpful, empathetic, and human-friendly than ever before.

#TuesdayTrivia #Robotic_Skin #Cambridge_University #Smart_skin_Revolution #RnD_Cell #CCA"""
    }
]


    return json.dumps(examples, indent=2)


@tool
def get_example_posts_for_critic() -> str:
    """Get example posts with quality scores."""
    examples = [
        {
            "title": "Researchers Have Developed a Cutting-Edge Silicon Photonic Chip to Enhance AI Efficiency",
            "content": """Researchers Have Developed a Cutting-Edge Silicon Photonic Chip to Enhance AI Efficiency

Researchers at the University of Florida have developed a silicon photonic chip that uses light to perform convolution operations, the pattern-recognition tasks at the heart of artificial intelligence. By encoding data as laser light and routing it through microscopic Fresnel lenses etched onto silicon, the chip processes information optically before converting it back into digital signals.
In testing, the system classified handwritten digits with roughly 98% accuracy, matching traditional electronic processors while consuming significantly less energy. The chip also supports wavelength multiplexing, allowing multiple colored lasers to handle parallel data streams, marking the first on-chip optical computation applied directly to neural networks.

#TuesdayTrivia #CCA #RnDCell #ArtificialIntelligence #SiliconPhotonics #OpticalComputing #AIHardware #EnergyEfficientAI""",
            "score": 8.5
        },
        {
            "title": "Researchers Have Developed Injectable Skin That Brings Hope for Burn Victims and Scar-Free Healing",
            "content": """Researchers Have Developed Injectable Skin That Brings Hope for Burn Victims and Scar-Free Healing

Researchers at the University of Linköping in Sweden have created an injectable gel filled with living skin cells that can be injected into wounds or 3D printed for skin transplants. Unlike conventional treatments that repair only the outer skin layer, this method rebuilds deeper dermal layers responsible for strength, elasticity, and blood vessel formation.
The material combines fibroblast cells with gelatin beads and a hyaluronic acid-based gel using click chemistry, allowing it to flow easily from a syringe before stabilizing in place. In animal studies, the cells survived, formed new blood vessels, and promoted healthy skin regeneration, opening new possibilities for scar-free healing and future organ repair.

#TuesdayTrivia #CCA #RnDCell #RegenerativeMedicine #TissueEngineering #SkinInASyringe #3DPrintedSkin #WoundHealing""",
            "score": 9.0
        },
        {
            "title": "Researchers Have Developed AI-Engineered Immune Cells to Target Cancer in Record Time",
            "content": """Researchers Have Developed AI-Engineered Immune Cells to Target Cancer in Record Time

Researchers have developed an AI-driven approach that rapidly reprograms human immune cells to recognize and destroy cancer cells within weeks instead of months. By applying machine-learning models, the team designed highly specific T-cell receptors that accurately target cancer while minimizing damage to healthy tissue.
The AI system analyzed millions of possible receptor combinations to identify the most effective designs, enabling personalized immune therapies tailored to each patient’s cancer type. This approach could significantly accelerate treatment development and move cancer therapy beyond one-size-fits-all solutions.

#TuesdayTrivia #CCA #RnDCell #CancerResearch #Immunotherapy #ArtificialIntelligence #PrecisionMedicine #FutureOfMedicine""",
            "score": 9.5
        }
    ]

    return json.dumps(examples, indent=2)


# --- 6. WORKFLOW & STATE ---

class WorkflowStage(Enum):
    IDLE = "idle"
    DISCOVERY = "discovery"
    CHECKPOINT_1 = "checkpoint_1"
    CURATOR = "curator"
    CHECKPOINT_2 = "checkpoint_2"
    WRITER = "writer"
    CRITIC = "critic"
    CHECKPOINT_3 = "checkpoint_3"
    FINALIZE = "finalize"
    COMPLETE = "complete"
    ERROR = "error"

class EnhancedAgentState(TypedDict):
    stage: str
    search_topic: str
    candidates: List[Dict]
    selected_story: Dict
    draft_summary: str
    quality_score: float
    critic_feedback: str
    retry_count: int
    error_message: str
    progress_log: List[str]

# --- 7. AGENT FUNCTIONS WITH MODEL LOGGING ---

def run_discovery(state: EnhancedAgentState, progress_callback=None) -> EnhancedAgentState:
    """Discovery Agent - Uses mistral-small-latest"""
    try:
        print("\n" + "="*70)
        print("🤖 DISCOVERY AGENT")
        print(f"📊 Model: mistral-small-latest")
        print(f"🎯 Purpose: Search and find scientific breakthroughs")
        print("="*70)
        
        if progress_callback:
            progress_callback("🔍 Discovery Agent (mistral-small) searching...")
        
        topic = state.get("search_topic", "general science")
        
        system_msg = SystemMessage(content=f"""You are the Discovery Agent for Tuesday Trivia.

REQUIREMENTS:
1. Use search_science_breakthroughs to find recent articles
2. Find 10-15 RECENT breakthroughs (last 1-4 weeks)
3. Focus on single discovery or breakthroughs posts instead of multiple ones like top 10 or top 5 posts
4. Avoid any awards and focus more on new breathroughs and discoveries

TITLE FOCUS:
1. MUST check TITLE similarity using check_topic_similarity - focus on the TITLE (first line) ONLY
3. When using check_topic_similarity, it specifically checks title word overlap
4. REJECT stories with titles that have >80% word overlap with existing titles

Output Format:
**Title:** [Unique, compelling title]
**Description:** [2-3 sentences]
**URL:** [Source link]
**Why Interesting:** [1 sentence hook]
---""")
        
        user_msg = HumanMessage(content=f"Search for recent breakthroughs in {topic}.")
        
        discovery_llm = llm_small.bind_tools([search_science_breakthroughs, get_all_previous_posts, check_topic_similarity])
        response = discovery_llm.invoke([system_msg, user_msg])
        conversation = [system_msg, user_msg, response]
        
        max_steps = 8
        steps = 0
        
        while hasattr(response, 'tool_calls') and response.tool_calls and steps < max_steps:
            tool_messages = []
            for tool_call in response.tool_calls:
                name = tool_call['name']
                print(f"🔧 Tool: {name}")
                if name == 'search_science_breakthroughs': 
                    res = search_science_breakthroughs.invoke(tool_call['args'])
                elif name == 'get_all_previous_posts': 
                    res = get_all_previous_posts.invoke(tool_call['args'])
                elif name == 'check_topic_similarity': 
                    res = check_topic_similarity.invoke(tool_call['args'])
                else: 
                    res = f"Unknown tool: {name}"
                tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call['id']))
            conversation.extend(tool_messages)
            response = discovery_llm.invoke(conversation)
            conversation.append(response)
            steps += 1
        
        print("✅ Discovery complete")
        state["candidates"] = [{"raw": response.content}]
        state["stage"] = WorkflowStage.CHECKPOINT_1.value
        state["progress_log"].append("✅ Discovery (mistral-small-latest)")
        return state
    except Exception as e:
        state["stage"] = WorkflowStage.ERROR.value
        state["error_message"] = f"Discovery failed: {str(e)}"
        return state

def run_curator(state: EnhancedAgentState, progress_callback=None) -> EnhancedAgentState:
    """Curator Agent - Uses mistral-small-latest"""
    try:
        print("\n" + "="*70)
        print("🤖 CURATOR AGENT")
        print(f"📊 Model: mistral-small-latest")
        print(f"🎯 Purpose: Rank candidates and select best story")
        print("="*70)
        
        if progress_callback:
            progress_callback("🎯 Curator (mistral-small) selecting story...")
        
        candidates = state.get("candidates", [])
        candidates_text = candidates[0].get("raw", "") if candidates else ""
        
        system_msg = SystemMessage(content="""You are the Curator Agent.

Requirements:
1. Use check_topic_similarity to verify the selected story's TITLE is unique
2. TITLES must have <80% word overlap with existing posts
3. Prioritize stories with completely unique, fresh titles
4. Avoid any awards ore recognition posts and focus more on new breathroughs and discoveries
5. Focus on single discovery or breakthroughs posts 
6. Reject posts which have multiple discoveries like top 5 top 10

Then rank on:
- Recency (1-10): How recent is the discovery?
- Significance (1-10): Scientific impact
- Engagement (1-10): Public interest potential

Output: RANKED CANDIDATES, then SELECTED STORY with title uniqueness check.""")
        
        if retry_count > 0:
            instruction = f"Refine the previous draft based on this feedback:\n{state.get('critic_feedback')}\n\nOriginal Story:\n{story_text}"
        else:
            instruction = f"Write Tuesday Trivia post based on:\n{story_text}"
        
        user_msg = HumanMessage(content=instruction)
        
        curator_llm = llm_medium.bind_tools([check_topic_similarity, get_all_previous_posts])
        response = curator_llm.invoke([system_msg, user_msg])
        conversation = [system_msg, user_msg, response]
        
        # Allow tool usage
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call['name']
                print(f"🔧 Tool: {name}")
                if name == 'check_topic_similarity':
                    res = check_topic_similarity.invoke(tool_call['args'])
                elif name == 'get_all_previous_posts':
                    res = get_all_previous_posts.invoke(tool_call['args'])
                else:
                    res = "Unknown tool"
                conversation.append(ToolMessage(content=str(res), tool_call_id=tool_call['id']))
            response = curator_llm.invoke(conversation)
        
        print("✅ Curation complete")
        state["selected_story"] = {"raw": response.content}
        state["stage"] = WorkflowStage.CHECKPOINT_2.value
        state["progress_log"].append("✅ Curation (mistral-large-latest)")
        return state
    except Exception as e:
        state["stage"] = WorkflowStage.ERROR.value
        state["error_message"] = f"Curation failed: {str(e)}"
        return state

def run_writer(state: EnhancedAgentState, progress_callback=None) -> EnhancedAgentState:
    """Writer Agent - Uses mistral-medium-latest"""
    try:
        print("\n" + "="*70)
        print("🤖 WRITER AGENT")
        print(f"📊 Model: mistral-medium-latest")
        print(f"🎯 Purpose: Write engaging Tuesday Trivia post")
        print("="*70)
        
        if progress_callback:
            progress_callback("✍️ Writer (mistral-medium) creating post...")
        
        selected_story = state.get("selected_story", {})
        story_text = selected_story.get("raw", "")
        retry_count = state.get("retry_count", 0)
        retry_context = f"\nPrevious feedback: {state.get('critic_feedback')}" if retry_count > 0 else ""

        system_msg = SystemMessage(content="""You are the Writer Agent for Tuesday Trivia.

CRITICAL FORMAT:
Title
[blank line]
Paragraph 1
[no blank line]
Paragraph 2
[blank line]
#TuesdayTrivia #RnDCell #CCA #Topic1 #Topic2

TITLE REQUIREMENTS:
- Use get_example_posts_for_writer to make almost same titles
- No catchy titles
- Title should begin like these options:
  - Researchers Have Developed AI-Engineered Immune Cells to Target Cancer in Record Time
  - Researchers Have Developed Injectable Skin That Brings Hope for Burn Victims and Scar-Free Healing
  - Scientists Have Discovered a new Exoplanet in a Distant Galaxy

CONTENT: 
- 140-180 words, technical but accessible
- use get_example_posts_for_writer to get example posts and write in a similar tone
- the beginning should be similar to these lines:
    - Researchers have developed
    - Scientists at the _ have discovered
    - Reasearchers at the _ have made a groundbreaking discovery
- keep the content formal as in the examples
HASHTAGS: Always include #TuesdayTrivia #RnDCell #CCA + 2-4 topic hashtags""")
        
        user_msg = HumanMessage(content=f"Write Tuesday Trivia post:\n{story_text}\n{retry_context}")
        
        writer_llm = llm_medium.bind_tools([get_example_posts_for_writer, check_topic_similarity, fetch_article_content, count_words])
        response = writer_llm.invoke([system_msg, user_msg])
        conversation = [system_msg, user_msg, response]
        
        steps = 0
        while hasattr(response, 'tool_calls') and response.tool_calls and steps < 5:
            for tool_call in response.tool_calls:
                name = tool_call['name']
                print(f"🔧 Tool: {name}")
                if name == 'get_example_posts_for_writer': res = get_example_posts_for_writer.invoke(tool_call['args'])
                elif name == 'check_topic_similarity': res = check_topic_similarity.invoke(tool_call['args'])
                elif name == 'fetch_article_content': res = fetch_article_content.invoke(tool_call['args'])
                elif name == 'count_words': res = count_words.invoke(tool_call['args'])
                else: res = "Unknown"
                conversation.append(ToolMessage(content=str(res), tool_call_id=tool_call['id']))
            response = writer_llm.invoke(conversation)
            conversation.append(response)
            steps += 1

        print("✅ Writing complete")
        state["draft_summary"] = response.content
        state["retry_count"] = retry_count + 1
        state["stage"] = WorkflowStage.CRITIC.value
        state["progress_log"].append(f"✅ Writing (mistral-medium-latest, attempt {retry_count + 1})")
        return state
    except Exception as e:
        state["stage"] = WorkflowStage.ERROR.value
        state["error_message"] = f"Writing failed: {str(e)}"
        return state

def run_critic(state: EnhancedAgentState, progress_callback=None) -> EnhancedAgentState:
    """Critic Agent - Uses mistral-large-latest with Auto-Loop"""
    try:
        print("\n" + "="*70)
        print("🤖 CRITIC AGENT")
        print(f"📊 Model: mistral-large-latest")
        print(f"🎯 Purpose: Evaluate and Auto-Correct")
        print("="*70)
        
        if progress_callback:
            progress_callback("🔍 Critic (mistral-large) evaluating...")
        
        draft = state.get("draft_summary", "")
        current_retries = state.get("retry_count", 0)
        MAX_RETRIES = 3
        
        system_msg = SystemMessage(content="""You are the Critic Agent for Tuesday Trivia.

EVALUATION CRITERIA:

1. TITLE  (2 points):
   - Title should be very close to example posts , use get_example_posts_for_critic for this
   - No catchy titles

2. FORMAT (2 points):
   - Title on first line
   - One blank line after title
   - Two paragraphs (no blank between them)
   - One blank line before hashtags
   - Hashtags include #TuesdayTrivia #RnDCell #CCA

3. CONTENT (3 points):
   - context
   - Technical details + impact
   - 140-180 words
   - Accurate information

4. STYLE (3 points):
   - writing style must be similar to example posts, use get_example_posts_for_critic for this
   - Technical concepts clear and accessible
   - must be formal as in the examples

OUTPUT FORMAT:
If Score < 7/10, you MUST provide a section called "ACTIONABLE FIXES" with specific instructions for the writer.

Structure your response exactly like this:
TOTAL SCORE: X/10
CRITIQUE: [General feedback]
ACTIONABLE FIXES: 
- [Fix 1]
- [Fix 2]
- [Fix 3]

""")
        
        user_msg = HumanMessage(content=f"Evaluate this post:\n\n{draft}")
        
        critic_llm = llm_large.bind_tools([get_example_posts_for_critic, check_topic_similarity])
        response = critic_llm.invoke([system_msg, user_msg])
        conversation = [system_msg, user_msg, response]
        
        steps = 0
        while hasattr(response, 'tool_calls') and response.tool_calls and steps < 5:
            for tool_call in response.tool_calls:
                name = tool_call['name']
                print(f"🔧 Tool: {name}")
                if name == 'get_example_posts_for_critic': 
                    res = get_example_posts_for_critic.invoke(tool_call['args'])
                elif name == 'check_topic_similarity':
                    res = check_topic_similarity.invoke(tool_call['args'])
                else:
                    res = "Unknown"
                conversation.append(ToolMessage(content=str(res), tool_call_id=tool_call['id']))
            
            # Get the agent's next thought/response
            response = critic_llm.invoke(conversation)
            conversation.append(response)
            steps += 1
        
        text = response.content or ""
        score = 5.0
        try:
            if "TOTAL SCORE:" in text:
                score = float(text.split("TOTAL SCORE:")[1].split("/")[0].strip())
        except: 
            pass
        
        print(f"✅ Evaluation complete - Score: {score}/10")
        
        # Save feedback for the writer to see
        state["quality_score"] = score
        state["critic_feedback"] = text
        state["progress_log"].append(f"✅ Evaluation (mistral-large-latest): {score}/10")

        # --- AUTOMATIC LOOP LOGIC ---
        # If score is low and we haven't retried too many times, send back to Writer
        if score < 8.0 and current_retries < MAX_RETRIES:
            print(f"🔄 Score too low ({score}). Auto-sending back to Writer with feedback...")
            state["stage"] = WorkflowStage.WRITER.value
        else:
            # Score is good OR we hit max retries -> Wait for human approval
            print("✅ Score acceptable or max retries reached. Waiting for human...")
            state["stage"] = WorkflowStage.CHECKPOINT_3.value

        return state
        
    except Exception as e:
        state["stage"] = WorkflowStage.ERROR.value
        state["error_message"] = f"Critic failed: {str(e)}"
        return state

def run_finalize(state: EnhancedAgentState, progress_callback=None) -> EnhancedAgentState:
    """Finalize and save to database"""
    try:
        print("\n" + "="*70)
        print("💾 FINALIZATION")
        print(f"🎯 Purpose: Save post to database and sync to HF")
        print("="*70)
        
        if progress_callback:
            progress_callback("💾 Saving to database and HF...")
        
        draft = state.get("draft_summary", "")
        score = state.get("quality_score", 0.0)
        
        topic = "Scientific Breakthrough"
        hashtags = ""
        lines = draft.split('\n')
        for line in lines:
            if "#" in line: 
                hashtags = line
                break
        
        res = save_approved_post.invoke({
            "topic": topic, 
            "summary": draft, 
            "source_url": "N/A", 
            "quality_score": score, 
            "hashtags": hashtags
        })
        
        print(f"✅ Post saved: {res}")
        state["stage"] = WorkflowStage.COMPLETE.value
        state["progress_log"].append(f"✅ Finalized and saved")
        return state
    except Exception as e:
        state["stage"] = WorkflowStage.ERROR.value
        state["error_message"] = f"Finalization failed: {str(e)}"
        return state

# --- 8. GRADIO INTERFACE ---

def create_initial_state(topic: str) -> EnhancedAgentState:
    return {
        "stage": WorkflowStage.IDLE.value,
        "search_topic": topic,
        "candidates": [],
        "selected_story": {},
        "draft_summary": "",
        "quality_score": 0.0,
        "critic_feedback": "",
        "retry_count": 0,
        "error_message": "",
        "progress_log": []
    }

def start_workflow(topic: str, progress=gr.Progress()):
    """Start the discovery process"""
    state = create_initial_state(topic)
    
    def update_progress(msg):
        progress(0.3, desc=msg)
    
    state = run_discovery(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    candidates_text = state["candidates"][0]["raw"] if state["candidates"] else "No candidates found"
    
    return (
        state,
        f"## 🔍 Discovery Results\n\n{candidates_text}",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=False),
        gr.update(visible=False)
    )

def handle_checkpoint1_approve(state, progress=gr.Progress()):
    """Handle approval at checkpoint 1"""
    def update_progress(msg):
        progress(0.5, desc=msg)
    
    state = run_curator(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    story_text = state["selected_story"]["raw"] if state["selected_story"] else "No story selected"
    
    return (
        state,
        f"## 🎯 Selected Story\n\n{story_text}\n\n**Optional:** Provide instructions in the textbox below if you want to pick a different story.",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=True),
        gr.update(visible=False)
    )

def handle_checkpoint1_reject(state, instructions, progress=gr.Progress()):
    """Handle rejection at checkpoint 1"""
    if instructions and instructions.strip():
        if "search_topic" in state:
            state["search_topic"] = f"{state['search_topic']} - Additional guidance: {instructions}"
    
    def update_progress(msg):
        progress(0.3, desc=msg)
    
    state = run_discovery(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    candidates_text = state["candidates"][0]["raw"] if state["candidates"] else "No candidates found"
    
    return (
        state,
        f"## 🔍 Discovery Results (New Search)\n\n{candidates_text}",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=True),
        gr.update(visible=False)
    )

def handle_checkpoint2_approve(state, progress=gr.Progress()):
    """Handle approval at checkpoint 2"""
    def update_progress(msg):
        progress(0.6, desc=msg)
    
    state = run_writer(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    progress(0.8, desc="Evaluating draft...")
    state = run_critic(state, lambda x: progress(0.8, desc=x))
    
    draft = state.get("draft_summary", "")
    score = state.get("quality_score", 0)
    feedback = state.get("critic_feedback", "")
    
    return (
        state,
        f"## ✍️ Draft Post\n\n{draft}\n\n---\n\n**Quality Score:** {score}/10\n\n**Feedback:**\n{feedback}",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=True),
        gr.update(visible=False)
    )

def handle_checkpoint2_different(state, instructions, progress=gr.Progress()):
    """Request different story"""
    if instructions and instructions.strip():
        if "selected_story" in state:
            state["critic_feedback"] = f"User preference: {instructions}"
    
    def update_progress(msg):
        progress(0.5, desc=msg)
    
    state = run_curator(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    story_text = state["selected_story"]["raw"] if state["selected_story"] else "No story selected"
    
    return (
        state,
        f"## 🎯 Selected Story (Alternative)\n\n{story_text}",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=True),
        gr.update(visible=False)
    )

def handle_checkpoint3_finalize(state, progress=gr.Progress()):
    """Finalize and save the post"""
    def update_progress(msg):
        progress(0.9, desc=msg)
    
    state = run_finalize(state, update_progress)
    
    draft = state.get("draft_summary", "")
    
    return (
        state,
        f"## ✅ Post Saved Successfully!\n\n{draft}\n\n---\n\n**Status:** Saved to database & HF Dataset\n**Quality Score:** {state.get('quality_score', 0)}/10",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        gr.update(visible=False),
        gr.update(visible=True)
    )

def handle_checkpoint3_edit(state, edit_instructions, progress=gr.Progress()):
    """Edit the draft based on instructions"""
    if not edit_instructions:
        return (
            state,
            "⚠️ Please provide edit instructions",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            edit_instructions,
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    if "critic_feedback" in state:
        state["critic_feedback"] += f"\n\nUser edit request: {edit_instructions}"
    
    def update_progress(msg):
        progress(0.6, desc=msg)
    
    state = run_writer(state, update_progress)
    
    if state["stage"] == WorkflowStage.ERROR.value:
        return (
            state,
            f"❌ **Error:** {state['error_message']}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    progress(0.8, desc="Re-evaluating draft...")
    state = run_critic(state, lambda x: progress(0.8, desc=x))
    
    draft = state.get("draft_summary", "")
    score = state.get("quality_score", 0)
    feedback = state.get("critic_feedback", "")
    
    return (
        state,
        f"## ✍️ Revised Draft\n\n{draft}\n\n---\n\n**Quality Score:** {score}/10\n\n**Feedback:**\n{feedback}",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(visible=True),
        gr.update(visible=False)
    )

def restart_workflow():
    """Reset everything"""
    return (
        None,
        "👋 Ready to start! Enter a topic and click 'Start Discovery'",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(visible=False),
        gr.update(visible=False)
    )

def copy_to_clipboard(state):
    """Return the final post for copying"""
    if state and state.get("draft_summary"):
        return state["draft_summary"]
    return "No post to copy"

# --- 9. GRADIO UI ---

css = """
.output-box {
    min-height: 400px;
    max-height: 600px;
    overflow-y: auto;
    padding: 20px;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    color: #000000 !important;
}
.output-box * {
    color: #000000 !important;
}
.output-box h1, .output-box h2, .output-box h3, .output-box h4 {
    color: #1a1a1a !important;
    font-weight: bold;
}
"""

with gr.Blocks(css=css, title="Tuesday Trivia Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧪 Tuesday Trivia Multi-Agent System
    
    **☁️ Cloud Storage:** All posts auto-sync to Hugging Face Datasets
    """)
    
    state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=2):
            output_display = gr.Markdown("👋 Ready to start! Enter a topic and click 'Start Discovery'", elem_classes="output-box")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Control Panel")
            
            topic_input = gr.Textbox(
                label="Search Topic",
                placeholder="e.g., quantum computing, biotechnology, space exploration",
                value="general science"
            )
            
            with gr.Row():
                start_btn = gr.Button("🚀 Start Discovery", variant="primary", visible=True)
                restart_btn = gr.Button("🔄 Start New", variant="secondary", visible=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📋 Decision Points")
            
            approve_btn = gr.Button("✅ Approve", variant="primary", visible=False)
            reject_btn = gr.Button("❌ Reject / Different", variant="stop", visible=False)
            
            edit_instructions = gr.Textbox(
                label="Edit Instructions (optional)",
                placeholder="Provide specific instructions for changes...",
                visible=False,
                lines=3
            )
            
            copy_btn = gr.Button("📋 Copy Final Post", variant="secondary", visible=False)
            copy_output = gr.Textbox(
                label="Post Content", 
                visible=False, 
                lines=10, 
                show_copy_button=True,  # <--- Adds the copy icon
                interactive=False       # <--- Makes it read-only
            )
            
            with gr.Accordion("☁️ Cloud Sync Status", open=False):
                sync_status = gr.Markdown(f"""
                **HF Dataset:** `{HF_DATASET_REPO}`  
                **Status:** {'✅ Connected' if HF_TOKEN else '❌ Not configured'}
                
                Posts auto-sync to HF after saving.
                """)
                
                manual_sync_btn = gr.Button("🔄 Manual Sync to HF", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("""
            ### ℹ️ Instructions
            
            **Workflow:**
            1. Discovery → Review candidates
            2. Curation → Review story selection
            3. Writing → Review draft
            4. Finalize → Copy & use!
            """)
    
    def manual_sync():
        result = sync_to_hf()
        if result:
            return "✅ Successfully synced to HF!"
        return "⚠️ Sync failed. Check logs."
    
    manual_sync_btn.click(fn=manual_sync, outputs=sync_status)
    
    # Event handlers
    start_btn.click(
        fn=start_workflow,
        inputs=[topic_input],
        outputs=[state, output_display, approve_btn, reject_btn, start_btn, restart_btn, edit_instructions, edit_instructions, copy_btn]
    )
    
    def smart_approve(s, instructions):
        if not s:
            return s, "No active workflow", gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update()
        
        stage = s.get("stage")
        if stage == WorkflowStage.CHECKPOINT_1.value:
            return handle_checkpoint1_approve(s)
        elif stage == WorkflowStage.CHECKPOINT_2.value:
            return handle_checkpoint2_approve(s)
        elif stage == WorkflowStage.CHECKPOINT_3.value:
            return handle_checkpoint3_finalize(s)
        
        return s, "Invalid stage", gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update()
    
    def smart_reject(s, instructions):
        if not s:
            return s, "No active workflow", gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update()
        
        stage = s.get("stage")
        if stage == WorkflowStage.CHECKPOINT_1.value:
            return handle_checkpoint1_reject(s, instructions)
        elif stage == WorkflowStage.CHECKPOINT_2.value:
            return handle_checkpoint2_different(s, instructions)
        elif stage == WorkflowStage.CHECKPOINT_3.value:
            return handle_checkpoint3_edit(s, instructions)
        
        return s, "Invalid stage", gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update()
    
    approve_btn.click(
        fn=smart_approve,
        inputs=[state, edit_instructions],
        outputs=[state, output_display, approve_btn, reject_btn, start_btn, restart_btn, edit_instructions, edit_instructions, copy_btn]
    )
    
    reject_btn.click(
        fn=smart_reject,
        inputs=[state, edit_instructions],
        outputs=[state, output_display, approve_btn, reject_btn, start_btn, restart_btn, edit_instructions, edit_instructions, copy_btn]
    )
    
    restart_btn.click(
        fn=restart_workflow,
        outputs=[state, output_display, approve_btn, reject_btn, start_btn, restart_btn, edit_instructions, edit_instructions, copy_btn]
    )
    
    copy_btn.click(
        fn=copy_to_clipboard,
        inputs=[state],
        outputs=[copy_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[copy_output]
    )

if __name__ == "__main__":
    demo.launch()