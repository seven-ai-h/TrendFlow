"""Seed the database with realistic sample data for dashboard demonstration."""
from database.db_setup import db_connection, getSession
from database.models import Story, Keyword, Article, PipelineRun
from datetime import datetime, timedelta
import random

random.seed(42)

PLATFORMS = ['hackernews', 'reddit', 'github', 'devto', 'rss']

HN_STORIES = [
    ("Show HN: I built a local LLM chat interface using Rust and WebAssembly", 847, 312),
    ("GPT-4o outperforms human experts on standardized coding benchmarks", 1203, 487),
    ("Ask HN: What's your current AI-assisted development workflow?", 634, 891),
    ("Mistral releases new 7B model that beats GPT-3.5 on most tasks", 982, 341),
    ("PostgreSQL 17 released: major performance improvements for OLAP workloads", 756, 198),
    ("Anthropic's Claude 3.5 Sonnet beats GPT-4 on 90% of coding tasks", 1456, 623),
    ("Python 3.13 final release: free-threaded mode and new JIT compiler", 891, 445),
    ("Show HN: Open-source alternative to Notion built with local-first architecture", 543, 267),
    ("GitHub Copilot now handles entire refactors across 50+ file codebases", 1102, 389),
    ("Kubernetes 1.30 – sidecar containers are now GA", 678, 156),
    ("Linear algebra library in pure Rust achieves NumPy-level performance", 445, 123),
    ("Building a RAG pipeline without LangChain: lessons from production", 892, 334),
    ("SQLite's new vector extension makes it viable for embedding search", 1034, 412),
    ("Ollama adds multi-GPU support and 2x inference speed improvements", 967, 278),
    ("Ask HN: Is anyone successfully using AI agents in production workflows?", 723, 1023),
]

REDDIT_STORIES = [
    ("DeepSeek R2 released – open weights, 671B parameters, MIT license", 4821, 892),
    ("Why I switched from Python to Rust for all my data pipelines", 3412, 567),
    ("Comparison: Claude vs GPT-4o vs Gemini on real engineering tasks", 5102, 1234),
    ("LLaMA 3.1 fine-tuned on code is now better than GPT-3.5 on most tasks", 2876, 445),
    ("TIL: PostgreSQL can replace Redis for most use cases with LISTEN/NOTIFY", 6234, 789),
    ("OpenAI's o1 model solves IMO gold-level math problems", 7891, 2341),
    ("My experience running a 70B model locally on consumer hardware", 3102, 678),
    ("GraphQL is dying and REST is winning again – discuss", 4567, 1890),
    ("Thoughts on the state of AI in 2025 – we're still early", 2134, 567),
    ("Docker Desktop alternative: Podman Desktop is now production-ready", 1892, 312),
]

GITHUB_STORIES = [
    ("ollama/ollama: Get up and running with large language models locally", 45231, 234),
    ("microsoft/promptflow: Build high-quality LLM apps", 12453, 89),
    ("langchain-ai/langchain: Build context-aware reasoning applications", 89234, 1023),
    ("huggingface/transformers: State-of-the-art ML for PyTorch, TF and JAX", 124567, 567),
    ("openai/whisper: Robust Speech Recognition via Large-Scale Weak Supervision", 67891, 234),
    ("gradio-app/gradio: Build ML demos and web apps with Python", 34521, 189),
    ("AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI", 156789, 892),
    ("ggerganov/llama.cpp: LLM inference in C/C++", 67891, 345),
    ("vllm-project/vllm: Easy, fast, cheap LLM serving for everyone", 23456, 156),
    ("deepseek-ai/DeepSeek-V3: DeepSeek-V3 model resources", 34512, 234),
]

DEVTO_STORIES = [
    ("Building a Production RAG System: Lessons from 6 Months in Production", 2341, 89),
    ("Why I Chose Rust Over Go for My High-Performance API", 1892, 67),
    ("A Complete Guide to Fine-tuning LLaMA 3 on Custom Data", 3102, 134),
    ("PostgreSQL Full-Text Search vs Elasticsearch: A Real Comparison", 1567, 45),
    ("Docker to Kubernetes: A Practical Migration Guide", 987, 34),
    ("Building Type-Safe APIs with Python and Pydantic v2", 1234, 56),
    ("LLM Prompt Engineering: Advanced Techniques That Actually Work", 4521, 178),
    ("Redis vs Valkey: Should You Migrate?", 2103, 89),
    ("Using Claude API to Build a Code Review Bot", 1789, 67),
    ("Svelte 5 Runes: A Deep Dive into the New Reactivity System", 2456, 112),
]

RSS_STORIES = [
    ("OpenAI Launches GPT-5 with 10x Longer Context Window", "TechCrunch"),
    ("Google DeepMind Claims AlphaCode 3 Surpasses Expert Programmers", "Wired"),
    ("The State of AI Safety: Researchers Warn of Emerging Capabilities", "MIT Tech Review"),
    ("Apple's M4 MacBook Pro Benchmarks Crush Previous Generation", "ArsTechnica"),
    ("Meta Open-Sources New Multimodal AI Model Trained on 1T Parameters", "The Verge"),
    ("Anthropic Raises $2B Series E at $15B Valuation", "TechCrunch"),
    ("Rust Becomes Second Official Language of the Linux Kernel", "ArsTechnica"),
    ("New Report: 70% of Fortune 500 Companies Now Use AI Coding Assistants", "Wired"),
    ("Quantum Computing Startup Achieves 1000-Qubit Milestone", "MIT Tech Review"),
    ("The Death of the Junior Developer? AI Pair Programming Changes Everything", "The Verge"),
]

# Tech entities with realistic frequencies across platforms
ENTITIES = {
    'hackernews': [
        ('llm', 45), ('python', 38), ('rust', 34), ('ai', 67), ('gpt', 29),
        ('claude', 23), ('kubernetes', 18), ('docker', 22), ('api', 41),
        ('open source', 31), ('machine learning', 28), ('rag', 19),
        ('vector', 17), ('embedding', 15), ('inference', 21),
        ('local llm', 14), ('fine tuning', 12), ('transformer', 16),
        ('postgresql', 24), ('sqlite', 18),
    ],
    'reddit': [
        ('ai', 89), ('llm', 67), ('openai', 54), ('claude', 41), ('gemini', 38),
        ('python', 72), ('rust', 45), ('gpt', 62), ('deep learning', 34),
        ('neural network', 28), ('fine tuning', 31), ('open source', 56),
        ('api', 48), ('local model', 29), ('benchmark', 37),
        ('rag', 22), ('agent', 41), ('copilot', 33), ('coding', 58),
        ('automation', 44),
    ],
    'github': [
        ('llm', 123), ('python', 98), ('ai', 145), ('transformer', 67),
        ('pytorch', 89), ('neural network', 56), ('fine tuning', 45),
        ('ollama', 78), ('langchain', 67), ('huggingface', 89),
        ('embedding', 56), ('vector', 67), ('rag', 45), ('inference', 78),
        ('model', 134), ('training', 89), ('cuda', 45), ('gpu', 67),
        ('diffusion', 34), ('multimodal', 28),
    ],
    'devto': [
        ('python', 67), ('javascript', 89), ('typescript', 76), ('api', 54),
        ('llm', 45), ('docker', 38), ('kubernetes', 34), ('react', 78),
        ('nextjs', 56), ('postgresql', 45), ('redis', 34), ('rust', 29),
        ('ai', 89), ('machine learning', 45), ('tutorial', 67),
        ('best practices', 34), ('performance', 45), ('testing', 38),
        ('devops', 29), ('cicd', 23),
    ],
    'rss': [
        ('ai', 78), ('openai', 56), ('google', 45), ('anthropic', 34),
        ('gpt', 67), ('llm', 45), ('startup', 34), ('funding', 23),
        ('technology', 89), ('machine learning', 56), ('chatbot', 34),
        ('enterprise', 45), ('cloud', 38), ('security', 29), ('data', 67),
        ('model', 78), ('research', 45), ('safety', 23), ('regulation', 18),
        ('investment', 29),
    ],
}


def seed():
    print("Initializing database…")
    db_connection()
    session = getSession()

    # Clear existing data
    session.query(Keyword).delete()
    session.query(Story).delete()
    session.query(Article).delete()
    session.query(PipelineRun).delete()
    session.commit()
    print("Cleared existing data")

    now = datetime.utcnow()

    # ── Stories ───────────────────────────────────────────────────────────────
    story_sets = [
        (HN_STORIES, 'hackernews', 'https://news.ycombinator.com/item?id={}'),
        (REDDIT_STORIES, 'reddit', 'https://reddit.com/r/programming/{}'),
        (GITHUB_STORIES, 'github', 'https://github.com/{}'),
        (DEVTO_STORIES, 'devto', 'https://dev.to/article/{}'),
    ]

    for stories, platform, url_tpl in story_sets:
        for i, item in enumerate(stories):
            title, score, comments = item
            age = timedelta(hours=random.randint(1, 48))
            session.add(Story(
                title=title,
                score=score,
                num_comments=comments,
                url=url_tpl.format(random.randint(10000, 99999)),
                platform=platform,
                timestamp=now - age,
            ))

    for i, (title, source) in enumerate(RSS_STORIES):
        age = timedelta(hours=random.randint(1, 24))
        session.add(Article(
            title=title,
            url=f"https://example.com/article/{i}",
            source=source,
            published_at=now - age,
            platform='rss',
            timestamp=now - age,
        ))
    session.commit()
    print(f"Seeded {sum(len(s) for s,*_ in story_sets)} stories + {len(RSS_STORIES)} articles")

    # ── Keywords across multiple days ─────────────────────────────────────────
    kw_count = 0
    for day_offset in range(7):
        base_time = now - timedelta(days=day_offset)
        for platform, kws in ENTITIES.items():
            for keyword, base_count in kws:
                # Add some realistic variance
                noise = random.uniform(0.6, 1.5)
                # Simulate a spike in the last 2 days for trending
                spike = 2.5 if day_offset < 2 and keyword in ('ai', 'llm', 'claude', 'rag', 'agent') else 1.0
                count = max(1, int(base_count * noise * spike))
                ts = base_time - timedelta(hours=random.randint(0, 23))
                session.add(Keyword(
                    keyword=keyword,
                    platform=platform,
                    count=count,
                    timestamp=ts,
                ))
                kw_count += 1

    session.commit()
    print(f"Seeded {kw_count} keyword records across 7 days")

    # ── Pipeline runs ─────────────────────────────────────────────────────────
    for i in range(5):
        start = now - timedelta(hours=i * 6)
        session.add(PipelineRun(
            started_at=start,
            finished_at=start + timedelta(seconds=random.randint(8, 25)),
            status='success',
            stories_collected=random.randint(40, 150),
            keywords_extracted=random.randint(80, 200),
            sources_run='hackernews,reddit,github,devto,rss',
        ))
    session.commit()
    print("Seeded 5 pipeline runs")
    print("\nDone! Refresh the dashboard.")


if __name__ == "__main__":
    seed()
