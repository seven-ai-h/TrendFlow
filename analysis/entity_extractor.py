import re
from collections import Counter

try:
    import spacy as _spacy
    _nlp = None

    def _get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = _spacy.load("en_core_web_sm")
            except OSError:
                return None
        return _nlp
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    _get_nlp = lambda: None

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english')) | {
    'hn', 'show', 'ask', 'tell', 'use', 'using', 'used', 'also', 'new', 'get',
    'make', 'made', 'need', 'want', 'way', 'like', 'just', 'one', 'would',
    'could', 'think', 'know', 'come', 'first', 'last', 'year', 'time', 'day',
    'thing', 'said', 'says', 'good', 'great', 'best', 'top', 'free', 'open',
}

# Known tech terms to boost (double-counted for emphasis)
TECH_TERMS = {
    'ai', 'ml', 'llm', 'llms', 'gpt', 'claude', 'gemini', 'openai', 'anthropic',
    'mistral', 'rag', 'rlhf', 'gpu', 'tpu', 'cuda', 'wasm', 'webassembly',
    'python', 'rust', 'golang', 'typescript', 'javascript', 'java', 'swift',
    'kubernetes', 'docker', 'k8s', 'aws', 'gcp', 'azure', 'terraform',
    'pytorch', 'tensorflow', 'jax', 'huggingface', 'langchain', 'vector',
    'embedding', 'transformer', 'diffusion', 'multimodal', 'finetuning',
    'react', 'vue', 'svelte', 'nextjs', 'fastapi', 'django', 'flask',
    'postgresql', 'mongodb', 'redis', 'kafka', 'spark', 'dbt', 'airflow',
    'nlp', 'cv', 'api', 'saas', 'oss', 'iot', 'blockchain', 'sqlite',
    'linux', 'macos', 'windows', 'android', 'ios', 'github', 'devops',
    'cicd', 'microservices', 'graphql', 'grpc', 'webrtc', 'sqlite',
}


def extract_entities(text: str, top_n: int = 10) -> list:
    """
    Extract named entities and key noun phrases from text.
    Uses spaCy NER + noun chunks if available; falls back to enhanced NLTK
    with known tech term boosting and bigram extraction.
    """
    entities = []
    nlp = _get_nlp() if SPACY_AVAILABLE else None

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'EVENT'):
                clean = ent.text.strip().lower()
                if len(clean) > 2 and clean not in STOP_WORDS:
                    entities.append(clean)
        for chunk in doc.noun_chunks:
            clean = chunk.root.text.strip().lower()
            if len(clean) > 2 and clean not in STOP_WORDS:
                entities.append(clean)
    else:
        # Token-level extraction with tech term boosting
        tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9\+\#\.]*\b', text)
        for tok in tokens:
            lower = tok.lower()
            if lower in TECH_TERMS:
                entities.append(lower)
                entities.append(lower)  # weight tech terms double
            elif len(lower) > 3 and lower not in STOP_WORDS:
                entities.append(lower)

    # Bigrams from filtered tokens (core improvement over bag-of-words)
    clean_tokens = [
        t.lower() for t in re.findall(r'\b[A-Za-z]{3,}\b', text)
        if t.lower() not in STOP_WORDS
    ]
    for a, b in zip(clean_tokens, clean_tokens[1:]):
        entities.append(f"{a} {b}")

    counts = Counter(entities)
    return [e for e, _ in counts.most_common(top_n)]
