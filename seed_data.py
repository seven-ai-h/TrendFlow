"""
Seed the database with realistic sample data for the dashboard + Model Lab.

Keywords are given ARCHETYPE trajectories over 14 days so the ML task
(will tomorrow's mentions jump >1.4x?) is genuinely learnable and the
different model families produce differentiated, meaningful results.
"""
import math
import random
from datetime import datetime, timedelta

from database.db_setup import db_connection, getSession
from database.models import Story, Keyword, Article, PipelineRun

random.seed(7)

DAYS = 14
PLATFORMS = ['hackernews', 'reddit', 'github', 'devto', 'rss']

# keyword -> (archetype, base_level)
KEYWORD_TRAJECTORIES = {
    # rising: steady growth (frequent next-day jumps early on)
    'ai agents': ('rising', 20), 'rag': ('rising', 15), 'claude': ('rising', 18),
    'llm': ('rising', 30), 'vector database': ('rising', 10), 'fine tuning': ('rising', 12),
    # spike: sudden bursts (sharp jumps -> positive labels)
    'deepseek': ('spike', 8), 'gpt': ('spike', 25), 'openai': ('spike', 20),
    'quantum': ('spike', 6), 'gemini': ('spike', 14),
    # falling: declining interest (mostly negatives)
    'blockchain': ('falling', 22), 'web3': ('falling', 18), 'nft': ('falling', 15),
    'metaverse': ('falling', 12),
    # cyclical: weekly ups and downs (mixed labels)
    'python': ('cyclical', 40), 'javascript': ('cyclical', 35), 'rust': ('cyclical', 25),
    'kubernetes': ('cyclical', 20), 'docker': ('cyclical', 22),
    # flat: stable (rare jumps)
    'postgresql': ('flat', 18), 'redis': ('flat', 14), 'linux': ('flat', 30),
    'api': ('flat', 45), 'sqlite': ('flat', 10),
    # volatile: noisy (unpredictable — separates strong models from weak)
    'transformer': ('volatile', 20), 'pytorch': ('volatile', 24),
    'embedding': ('volatile', 12), 'inference': ('volatile', 16),
    'multimodal': ('volatile', 9), 'agent': ('volatile', 28),
}


def trajectory_value(archetype: str, base: float, day: int, total: int) -> float:
    """Return the mention count for a keyword on a given day index (0=oldest)."""
    t = day / max(1, total - 1)  # 0..1
    if archetype == 'rising':
        val = base * (1 + 2.2 * t) * random.uniform(0.85, 1.15)
    elif archetype == 'falling':
        val = base * (1.8 - 1.5 * t) * random.uniform(0.85, 1.15)
    elif archetype == 'spike':
        # baseline low with 2-3 sharp spikes
        spike = 3.0 if day in (int(total * 0.4), int(total * 0.7), total - 2) else 1.0
        val = base * spike * random.uniform(0.8, 1.2)
    elif archetype == 'cyclical':
        val = base * (1 + 0.6 * math.sin(day * math.pi / 3.5)) * random.uniform(0.9, 1.1)
    elif archetype == 'flat':
        val = base * random.uniform(0.9, 1.1)
    else:  # volatile
        val = base * random.uniform(0.4, 2.2)
    return max(1, val)


# ── Story headlines (for the Stories tab) ────────────────────────────────────
HN_STORIES = [
    ("Show HN: Local LLM chat interface in Rust + WebAssembly", 847, 312),
    ("Claude 3.5 beats GPT-4 on 90% of coding tasks", 1456, 623),
    ("DeepSeek R2: open weights, 671B params, MIT license", 1203, 487),
    ("Python 3.13: free-threaded mode and a new JIT", 891, 445),
    ("Building a RAG pipeline without LangChain", 892, 334),
    ("SQLite vector extension makes embedding search viable", 1034, 412),
    ("Ollama adds multi-GPU support, 2x inference speed", 967, 278),
    ("PostgreSQL 17: big OLAP performance wins", 756, 198),
    ("Ask HN: Your current AI-assisted dev workflow?", 634, 891),
    ("AI agents in production: what actually works", 723, 1023),
]
REDDIT_STORIES = [
    ("DeepSeek R2 released — open weights, MIT license", 4821, 892),
    ("Claude vs GPT-4o vs Gemini on real engineering tasks", 5102, 1234),
    ("Why I switched from Python to Rust for data pipelines", 3412, 567),
    ("o1 solves IMO gold-level math problems", 7891, 2341),
    ("Running a 70B model locally on consumer hardware", 3102, 678),
    ("PostgreSQL can replace Redis for most use cases", 6234, 789),
]
GITHUB_STORIES = [
    ("ollama/ollama: run LLMs locally", 45231, 234),
    ("langchain-ai/langchain: context-aware reasoning apps", 89234, 1023),
    ("ggerganov/llama.cpp: LLM inference in C/C++", 67891, 345),
    ("vllm-project/vllm: fast, cheap LLM serving", 23456, 156),
    ("deepseek-ai/DeepSeek-V3: model resources", 34512, 234),
    ("huggingface/transformers: SOTA ML", 124567, 567),
]
DEVTO_STORIES = [
    ("Production RAG: 6 months of lessons", 2341, 89),
    ("Fine-tuning LLaMA 3 on custom data", 3102, 134),
    ("LLM prompt engineering that actually works", 4521, 178),
    ("Building a code review bot with the Claude API", 1789, 67),
    ("Type-safe APIs with Python and Pydantic v2", 1234, 56),
]
RSS_STORIES = [
    ("OpenAI Launches GPT-5 with 10x Longer Context", "TechCrunch"),
    ("DeepMind Claims AlphaCode 3 Surpasses Experts", "Wired"),
    ("Anthropic Raises $2B Series E at $15B Valuation", "TechCrunch"),
    ("Rust Becomes Second Official Linux Kernel Language", "ArsTechnica"),
    ("70% of Fortune 500 Now Use AI Coding Assistants", "Wired"),
    ("Quantum Startup Hits 1000-Qubit Milestone", "MIT Tech Review"),
]


def seed():
    print("Initializing database…")
    db_connection()
    session = getSession()

    session.query(Keyword).delete()
    session.query(Story).delete()
    session.query(Article).delete()
    session.query(PipelineRun).delete()
    session.commit()
    print("Cleared existing data")

    now = datetime.utcnow()

    # ── Keywords: archetype trajectories over DAYS days, split across platforms ─
    kw_rows = 0
    for keyword, (archetype, base) in KEYWORD_TRAJECTORIES.items():
        # each keyword lives on 2-4 platforms
        n_plat = random.randint(2, 4)
        kw_platforms = random.sample(PLATFORMS, n_plat)
        for day in range(DAYS):
            day_ts = now - timedelta(days=(DAYS - 1 - day))
            total = trajectory_value(archetype, base, day, DAYS)
            # distribute the day's total across this keyword's platforms
            weights = [random.uniform(0.5, 1.5) for _ in kw_platforms]
            wsum = sum(weights)
            for plat, w in zip(kw_platforms, weights):
                count = max(1, int(total * w / wsum))
                ts = day_ts - timedelta(hours=random.randint(0, 20))
                session.add(Keyword(
                    keyword=keyword, platform=plat, count=count, timestamp=ts))
                kw_rows += 1
    session.commit()
    print(f"Seeded {kw_rows} keyword records "
          f"({len(KEYWORD_TRAJECTORIES)} keywords x {DAYS} days)")

    # ── Stories ───────────────────────────────────────────────────────────────
    story_sets = [
        (HN_STORIES, 'hackernews', 'https://news.ycombinator.com/item?id={}'),
        (REDDIT_STORIES, 'reddit', 'https://reddit.com/r/programming/{}'),
        (GITHUB_STORIES, 'github', 'https://github.com/{}'),
        (DEVTO_STORIES, 'devto', 'https://dev.to/a/{}'),
    ]
    n_stories = 0
    for stories, platform, tpl in story_sets:
        for title, score, comments in stories:
            session.add(Story(
                title=title, score=score, num_comments=comments,
                url=tpl.format(random.randint(10000, 99999)),
                platform=platform,
                timestamp=now - timedelta(hours=random.randint(1, 60))))
            n_stories += 1
    for i, (title, source) in enumerate(RSS_STORIES):
        ts = now - timedelta(hours=random.randint(1, 40))
        session.add(Article(
            title=title, url=f"https://example.com/a/{i}", source=source,
            published_at=ts, platform='rss', timestamp=ts))
    session.commit()
    print(f"Seeded {n_stories} stories + {len(RSS_STORIES)} articles")

    # ── Pipeline runs ─────────────────────────────────────────────────────────
    for i in range(6):
        start = now - timedelta(hours=i * 5)
        session.add(PipelineRun(
            started_at=start,
            finished_at=start + timedelta(seconds=random.randint(9, 28)),
            status='success',
            stories_collected=random.randint(40, 160),
            keywords_extracted=random.randint(90, 220),
            sources_run='hackernews,reddit,github,devto,rss'))
    session.commit()
    print("Seeded 6 pipeline runs")
    print("\nDone! Restart / refresh the dashboard.")


if __name__ == "__main__":
    seed()
