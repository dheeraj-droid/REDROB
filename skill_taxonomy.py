"""
Semantic skill taxonomy for the Redrob AI Candidate Ranking System.

Maps individual skill names to categories, enabling semantic matching
rather than keyword matching. e.g., "Milvus" → vector_db_retrieval.

Also provides relevance weights per category for the Senior AI Engineer role.
"""

# ---------------------------------------------------------------------------
# Skill → Category mapping
# ---------------------------------------------------------------------------
# Each skill is mapped to one primary category. Categories are scored
# differently depending on the target JD.

SKILL_TO_CATEGORY = {
    # ── Core AI / ML ──────────────────────────────────────────────────────
    "NLP":                        "core_ai_ml",
    "Natural Language Processing": "core_ai_ml",
    "Fine-tuning LLMs":           "core_ai_ml",
    "LLM Fine-tuning":            "core_ai_ml",
    "LoRA":                       "core_ai_ml",
    "QLoRA":                      "core_ai_ml",
    "PEFT":                       "core_ai_ml",
    "Transformers":               "core_ai_ml",
    "BERT":                       "core_ai_ml",
    "GPT":                        "core_ai_ml",
    "Hugging Face":               "core_ai_ml",
    "LangChain":                  "core_ai_ml",
    "Prompt Engineering":         "core_ai_ml",
    "RAG":                        "core_ai_ml",
    "Retrieval Augmented Generation": "core_ai_ml",
    "Sentence Transformers":      "core_ai_ml",
    "Embeddings":                 "core_ai_ml",
    "Word Embeddings":            "core_ai_ml",
    "Text Classification":        "core_ai_ml",
    "Named Entity Recognition":   "core_ai_ml",
    "Sentiment Analysis":         "core_ai_ml",
    "Information Retrieval":      "core_ai_ml",
    "Search Ranking":             "core_ai_ml",
    "Recommendation Systems":     "core_ai_ml",
    "Learning to Rank":           "core_ai_ml",
    "Ranking Systems":            "core_ai_ml",

    # ── ML Foundations ────────────────────────────────────────────────────
    "Machine Learning":           "ml_foundations",
    "Deep Learning":              "ml_foundations",
    "Neural Networks":            "ml_foundations",
    "PyTorch":                    "ml_foundations",
    "TensorFlow":                 "ml_foundations",
    "Keras":                      "ml_foundations",
    "Scikit-learn":               "ml_foundations",
    "XGBoost":                    "ml_foundations",
    "LightGBM":                   "ml_foundations",
    "Random Forest":              "ml_foundations",
    "Statistical Modeling":       "ml_foundations",
    "Feature Engineering":        "ml_foundations",
    "Model Evaluation":           "ml_foundations",
    "A/B Testing":                "ml_foundations",
    "Experiment Design":          "ml_foundations",
    "Data Science":               "ml_foundations",
    "Predictive Modeling":        "ml_foundations",
    "Regression":                 "ml_foundations",
    "Classification":             "ml_foundations",
    "Clustering":                 "ml_foundations",

    # ── Computer Vision (adjacent but not core for this role) ─────────────
    "Image Classification":       "computer_vision",
    "Object Detection":           "computer_vision",
    "Image Segmentation":         "computer_vision",
    "OpenCV":                     "computer_vision",
    "YOLO":                       "computer_vision",
    "GANs":                       "computer_vision",
    "Computer Vision":            "computer_vision",
    "CNN":                        "computer_vision",
    "Stable Diffusion":           "computer_vision",
    "Image Generation":           "computer_vision",

    # ── Speech / Audio (domain mismatch per JD) ──────────────────────────
    "Speech Recognition":         "speech_audio",
    "TTS":                        "speech_audio",
    "Text to Speech":             "speech_audio",
    "ASR":                        "speech_audio",
    "Audio Processing":           "speech_audio",
    "Voice AI":                   "speech_audio",

    # ── Vector DB / Retrieval Infrastructure ──────────────────────────────
    "Milvus":                     "vector_db_retrieval",
    "Pinecone":                   "vector_db_retrieval",
    "Weaviate":                   "vector_db_retrieval",
    "Qdrant":                     "vector_db_retrieval",
    "ChromaDB":                   "vector_db_retrieval",
    "FAISS":                      "vector_db_retrieval",
    "Elasticsearch":              "vector_db_retrieval",
    "OpenSearch":                 "vector_db_retrieval",
    "Vector Databases":           "vector_db_retrieval",
    "Semantic Search":            "vector_db_retrieval",
    "Hybrid Search":              "vector_db_retrieval",
    "BM25":                       "vector_db_retrieval",
    "Solr":                       "vector_db_retrieval",
    "Lucene":                     "vector_db_retrieval",

    # ── MLOps / Model Serving ─────────────────────────────────────────────
    "MLflow":                     "mlops",
    "Weights & Biases":           "mlops",
    "W&B":                        "mlops",
    "BentoML":                    "mlops",
    "Kubeflow":                   "mlops",
    "TFX":                        "mlops",
    "Model Serving":              "mlops",
    "ONNX":                       "mlops",
    "TensorRT":                   "mlops",
    "Triton":                     "mlops",
    "SageMaker":                  "mlops",
    "Vertex AI":                  "mlops",

    # ── Python Ecosystem ──────────────────────────────────────────────────
    "Python":                     "python_ecosystem",
    "Flask":                      "python_ecosystem",
    "FastAPI":                    "python_ecosystem",
    "Django":                     "python_ecosystem",
    "Pandas":                     "python_ecosystem",
    "NumPy":                      "python_ecosystem",
    "Matplotlib":                 "python_ecosystem",
    "Streamlit":                  "python_ecosystem",
    "Jupyter":                    "python_ecosystem",

    # ── Data Engineering ──────────────────────────────────────────────────
    "Spark":                      "data_engineering",
    "PySpark":                    "data_engineering",
    "Apache Spark":               "data_engineering",
    "Airflow":                    "data_engineering",
    "Apache Airflow":             "data_engineering",
    "Kafka":                      "data_engineering",
    "Apache Kafka":               "data_engineering",
    "Apache Beam":                "data_engineering",
    "Apache Flink":               "data_engineering",
    "Databricks":                 "data_engineering",
    "Snowflake":                  "data_engineering",
    "dbt":                        "data_engineering",
    "ETL":                        "data_engineering",
    "Data Pipelines":             "data_engineering",
    "SQL":                        "data_engineering",
    "PostgreSQL":                 "data_engineering",
    "MongoDB":                    "data_engineering",
    "Redis":                      "data_engineering",
    "Hadoop":                     "data_engineering",

    # ── Cloud & Infrastructure ────────────────────────────────────────────
    "AWS":                        "cloud_infra",
    "GCP":                        "cloud_infra",
    "Azure":                      "cloud_infra",
    "Docker":                     "cloud_infra",
    "Kubernetes":                 "cloud_infra",
    "Terraform":                  "cloud_infra",
    "CI/CD":                      "cloud_infra",
    "Linux":                      "cloud_infra",
    "Git":                        "cloud_infra",

    # ── Frontend / Web (not relevant for this role) ───────────────────────
    "React":                      "frontend_web",
    "Angular":                    "frontend_web",
    "Vue.js":                     "frontend_web",
    "JavaScript":                 "frontend_web",
    "TypeScript":                 "frontend_web",
    "Node.js":                    "frontend_web",
    "HTML":                       "frontend_web",
    "CSS":                        "frontend_web",
    "Tailwind":                   "frontend_web",
    "Next.js":                    "frontend_web",
    "Redux":                      "frontend_web",
    "GraphQL":                    "frontend_web",
    "REST API":                   "frontend_web",
    "Webpack":                    "frontend_web",

    # ── Business / Non-Technical ──────────────────────────────────────────
    "Project Management":         "business_nontechnical",
    "Agile":                      "business_nontechnical",
    "Scrum":                      "business_nontechnical",
    "JIRA":                       "business_nontechnical",
    "Marketing":                  "business_nontechnical",
    "SEO":                        "business_nontechnical",
    "Content Writing":            "business_nontechnical",
    "PowerPoint":                 "business_nontechnical",
    "Excel":                      "business_nontechnical",
    "Photoshop":                  "business_nontechnical",
    "Figma":                      "business_nontechnical",
    "SAP":                        "business_nontechnical",
    "Six Sigma":                  "business_nontechnical",
    "Accounting":                 "business_nontechnical",
    "Supply Chain":               "business_nontechnical",
    "CRM":                        "business_nontechnical",
    "Salesforce":                 "business_nontechnical",
    "Tableau":                    "business_nontechnical",
    "Power BI":                   "business_nontechnical",
}

# ---------------------------------------------------------------------------
# Category relevance weights for "Senior AI Engineer — Founding Team"
# ---------------------------------------------------------------------------
# Higher = more relevant to the role. Range: 0.0 to 1.0

CATEGORY_RELEVANCE = {
    "core_ai_ml":            1.0,   # Must-have: embeddings, NLP, retrieval, ranking
    "vector_db_retrieval":   1.0,   # Must-have: production vector DB experience
    "ml_foundations":         0.8,   # Strong signal: ML fundamentals
    "mlops":                 0.7,   # Nice-to-have: model deployment, monitoring
    "python_ecosystem":      0.6,   # Expected: strong Python
    "data_engineering":      0.5,   # Adjacent: useful but not core
    "cloud_infra":           0.4,   # Adjacent: operational skills
    "computer_vision":       0.15,  # Mild negative signal per JD unless paired with NLP
    "speech_audio":          0.10,  # Negative signal per JD: domain mismatch
    "frontend_web":          0.05,  # Irrelevant for this role
    "business_nontechnical": 0.0,   # Irrelevant — and may indicate non-technical profile
}

# ---------------------------------------------------------------------------
# JD "Must-Have" skill clusters — at least 1 skill from each cluster needed
# ---------------------------------------------------------------------------
MUST_HAVE_CLUSTERS = {
    "embeddings_retrieval": {
        "skills": [
            "Sentence Transformers", "Embeddings", "Word Embeddings",
            "NLP", "BERT", "Transformers", "Information Retrieval",
            "Semantic Search", "RAG", "Retrieval Augmented Generation",
            "Search Ranking", "Ranking Systems", "Learning to Rank",
            "Recommendation Systems", "BM25", "Hybrid Search",
        ],
        "weight": 1.0,
        "description": "Production embeddings / retrieval experience",
    },
    "vector_db": {
        "skills": [
            "Milvus", "Pinecone", "Weaviate", "Qdrant", "ChromaDB",
            "FAISS", "Elasticsearch", "OpenSearch", "Solr", "Lucene",
            "Vector Databases", "Semantic Search", "Hybrid Search",
        ],
        "weight": 1.0,
        "description": "Vector DB / hybrid search infrastructure",
    },
    "python": {
        "skills": [
            "Python", "Flask", "FastAPI", "Django", "Pandas", "NumPy",
            "Streamlit", "Jupyter",
        ],
        "weight": 0.8,
        "description": "Strong Python",
    },
}

# ---------------------------------------------------------------------------
# Proficiency weights
# ---------------------------------------------------------------------------
PROFICIENCY_WEIGHT = {
    "expert":       1.0,
    "advanced":     0.8,
    "intermediate": 0.5,
    "beginner":     0.2,
}

# ---------------------------------------------------------------------------
# Consulting / Services companies (used for career-quality scoring)
# ---------------------------------------------------------------------------
CONSULTING_SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "ltimindtree", "persistent systems", "hexaware", "cyient",
    "zensar", "niit", "coforge", "birlasoft", "sonata software",
    # Fictional companies from dataset
    "dunder mifflin", "initech", "acme corp", "globex inc",
    "stark industries", "wayne enterprises", "umbrella corp",
    "soylent corp", "hooli",
}

# ---------------------------------------------------------------------------
# Non-technical titles (strong signal of keyword-stuffer if paired with
# many AI skills)
# ---------------------------------------------------------------------------
NON_TECHNICAL_TITLES = {
    "hr manager", "marketing manager", "sales executive",
    "accountant", "content writer", "graphic designer",
    "operations manager", "customer support", "project manager",
    "business analyst", "civil engineer", "mechanical engineer",
    "supply chain manager", "procurement manager",
    "administrative assistant", "office manager",
}

# ---------------------------------------------------------------------------
# AI/ML-relevant titles
# ---------------------------------------------------------------------------
AI_ML_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer",
    "data scientist", "senior data scientist",
    "research scientist", "applied scientist",
    "nlp engineer", "deep learning engineer",
    "computer vision engineer", "mlops engineer",
    "ai researcher", "ml researcher",
    "junior ml engineer", "junior ai engineer",
    "lead ai engineer", "lead ml engineer",
    "principal engineer", "staff engineer",
    "backend engineer",  # can be relevant if description shows AI work
}

# Title-to-relevance mapping for this specific JD
TITLE_RELEVANCE = {
    "ai engineer":                     1.0,
    "senior ai engineer":              1.0,
    "ml engineer":                     1.0,
    "senior ml engineer":              1.0,
    "machine learning engineer":       1.0,
    "senior machine learning engineer": 1.0,
    "junior ml engineer":              0.7,
    "junior ai engineer":              0.7,
    "data scientist":                  0.8,
    "senior data scientist":           0.85,
    "research scientist":              0.6,
    "applied scientist":               0.75,
    "nlp engineer":                    0.9,
    "deep learning engineer":          0.85,
    "mlops engineer":                  0.7,
    "backend engineer":                0.5,
    "software engineer":               0.45,
    "data engineer":                   0.45,
    "analytics engineer":              0.3,
    "devops engineer":                 0.2,
    "full stack developer":            0.15,
    # Non-technical titles
    "project manager":                 0.05,
    "business analyst":                0.1,
    "hr manager":                      0.0,
    "marketing manager":               0.0,
    "sales executive":                 0.0,
    "accountant":                      0.0,
    "content writer":                  0.0,
    "graphic designer":                0.0,
    "operations manager":              0.0,
    "customer support":                0.0,
    "civil engineer":                  0.02,
    "mechanical engineer":             0.02,
}


def classify_skill(skill_name: str) -> str:
    """Return the category for a skill name, or 'unknown'."""
    # Direct lookup
    if skill_name in SKILL_TO_CATEGORY:
        return SKILL_TO_CATEGORY[skill_name]

    # Case-insensitive fallback
    skill_lower = skill_name.lower().strip()
    for known_skill, category in SKILL_TO_CATEGORY.items():
        if known_skill.lower() == skill_lower:
            return category

    return "unknown"


def get_category_relevance(category: str) -> float:
    """Return the relevance weight for a skill category."""
    return CATEGORY_RELEVANCE.get(category, 0.2)


def get_title_relevance(title: str) -> float:
    """Return the relevance score for a job title."""
    title_lower = title.lower().strip()
    if title_lower in TITLE_RELEVANCE:
        return TITLE_RELEVANCE[title_lower]

    # Partial match fallback
    for known_title, score in TITLE_RELEVANCE.items():
        if known_title in title_lower or title_lower in known_title:
            return score

    return 0.1  # Unknown title — slight relevance


def is_non_technical_title(title: str) -> bool:
    """Check if a title is non-technical."""
    return title.lower().strip() in NON_TECHNICAL_TITLES


def is_consulting_company(company: str) -> bool:
    """Check if a company is a consulting/services firm."""
    return company.lower().strip() in CONSULTING_SERVICES_COMPANIES
