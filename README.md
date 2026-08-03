# 🏢 Corporate HR Assistant using RAG

A Streamlit RAG assistant that answers only from uploaded HR policy PDFs.
The pipeline is PDF loading → preprocessing → chunking → MiniLM embeddings →
ChromaDB retrieval → grounded OpenRouter generation.

## Important

The repository does **not** contain the HR policy PDFs. Add them in either way:

1. Put PDF files inside `documents/` before deployment, or
2. Upload them from the Streamlit sidebar and click **Save PDFs & Rebuild**.

## Local setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=qwen/qwen3-8b
```

Run:

```powershell
streamlit run streamlit_app.py
```

## Streamlit Cloud

Add these values under **App settings → Secrets**:

```toml
OPENROUTER_API_KEY = "your_key_here"
OPENROUTER_MODEL = "qwen/qwen3-8b"
```

Then either commit the required PDFs under `documents/`, or upload them from
inside the deployed app. Uploaded files and the generated Chroma database are
stored on the app instance and may need to be uploaded again after a redeploy
or instance reset.

## Main fixes in this version

- Adds the missing `langchain-openai` dependency.
- Makes `st.set_page_config` the first Streamlit command.
- Adds PDF upload and rebuild controls.
- Uses paths relative to the project directory, not the current shell folder.
- Rebuilds Chroma automatically when the PDF corpus changes.
- Persists successful assistant messages in session state.
- Returns the exact fallback without calling the LLM when retrieval is empty.
- Uses `OPENROUTER_MODEL` instead of ignoring the configured model.
- Corrects provider labels from Gemini to OpenRouter.
- Improves dark-theme readability in the sidebar and top header.

## Security

Never commit `.env` or `.streamlit/secrets.toml`. If a real API key has been
shared or committed, revoke it and create a new one.
