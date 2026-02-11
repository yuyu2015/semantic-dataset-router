# Semantic Dataset Router 🧭

A lightweight, production-ready RAG routing agent capable of dynamically selecting the most relevant CSV context based on user query embeddings.

## 🚀 Key Features
* **Zero-Shot Routing:** Uses `sentence-transformers` for semantic intent matching.
* **Cosine similarity + threshold:** Dataset descriptions are embedded; the query is routed to the dataset with the highest similarity score only when that score is above a configurable threshold.
* **Prompt design:** General instruction plus optional dataset context (included only when a dataset is selected).
* **Modern Stack:** Built with `uv`, `Typer`, and `Pydantic` for robust dependency and configuration management.

## How it works

1. **Dataset descriptions** for `shopping_habits.csv` and `weekly_searches_for_programming_languages.csv` are embedded once using a sentence transformer (default: `all-MiniLM-L6-v2`).
2. **User query** is embedded and compared to each dataset description via **cosine similarity**.
3. The **dataset with the larger score** is chosen only if the score is **≥ threshold**; otherwise no dataset context is attached.
4. **Prompt** is built as: general instruction + user question + (if above threshold) “Relevant dataset context” with the selected CSV preview.

## CLI

```bash
# Route a query (print similarity scores and selected dataset)
uv run python main.py route "How much do consumers spend on shopping?" -t 0.25

# Build full prompt with optional dataset context (for use with an LLM)
uv run python main.py query "Which programming language is most searched?" -t 0.25 -s

# Send prompt to OpenRouter and print the LLM answer (requires OPENROUTER_API_KEY)
uv run python main.py query "How much do consumers spend?" --call
uv run python main.py query "Python vs Java search trends?" -c -l openai/gpt-4o --scores
```

Options:
* `--threshold` / `-t`: Minimum cosine similarity to attach a dataset (default: 0.25).
* `--model` / `-m`: Sentence transformer model (default: `all-MiniLM-L6-v2`).
* `--instruction` / `-i`: General instruction text (query command only).
* `--scores` / `-s`: Show routing result before the prompt (query command only).
* `--call` / `-c`: Send the built prompt to OpenRouter and print the LLM response.
* `--llm-model` / `-l`: OpenRouter model id (default: `openai/gpt-4o-mini`).
* `--temperature`, `--max-tokens`: LLM sampling parameters.

**Configuration:** All environment variables are centralized in `config.py` via **Pydantic Settings** (`AppSettings`). Copy the example env file and edit with your values:

```bash
cp .env.example .env
# Windows: copy .env.example .env
```

Then set `OPENROUTER_API_KEY` (and any other overrides) in `.env`. The app loads `.env` from the project root automatically.

| Env var | Description | Default |
|---------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required for `--call`) | — |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `DEFAULT_LLM_MODEL` | Default OpenRouter model id | `openai/gpt-4o-mini` |
| `EMBEDDING_MODEL` | Sentence transformer for embeddings | `all-MiniLM-L6-v2` |
| `ROUTING_THRESHOLD` | Min cosine similarity to attach a dataset | `0.25` |

**General instructions:** Versioned prompt instructions live in `general_instructions.py` (`INSTRUCTIONS` dict and `DEFAULT_VERSION`). Use `--instruction-version` (e.g. `1`) to pick a version, or `--instruction` to pass custom text and ignore the versioned file.

**LLM (OpenRouter):** Set `OPENROUTER_API_KEY` in `.env` or the environment when using `--call`. The client in `llm_client.py` parses the response into a structured format (content, model, usage, finish_reason) and prints the answer plus usage in a readable layout.
