# Streamlit Repair Report

## Confirmed blockers

1. The project archive did not contain the `documents/` folder or any policy PDFs, while the app stopped immediately when no PDFs were present.
2. `07_prompting.py` imports `langchain_openai`, but `requirements.txt` did not install `langchain-openai`.
3. `st.secrets` was accessed before `st.set_page_config`.
4. The configured `OPENROUTER_MODEL` value was ignored.
5. Successful assistant replies were not added to `st.session_state.messages`.
6. Chroma used a working-directory-relative path, which can point to the wrong location during deployment.
7. The UI and README described Gemini although the runtime client used OpenRouter.
8. The dark theme did not explicitly style sidebar/header text, producing low-contrast text.
9. The uploaded project included live API credentials. They were removed from this repaired package.

## Repairs

- Added in-app PDF upload and rebuild workflow.
- Added an empty tracked `documents/` directory.
- Added the missing OpenRouter LangChain dependency.
- Moved page configuration before every other Streamlit call.
- Added project-relative paths.
- Added corpus fingerprinting and automatic Chroma rebuild when PDFs change.
- Added deterministic fallback without an LLM request when retrieval is empty.
- Persisted successful messages and sources in session state.
- Sanitized uploaded filenames.
- Added safe `.env.example` and Streamlit Secrets example files.
- Corrected provider names and theme contrast.

## Validation completed

- All Python files pass compilation and AST parsing.
- Secret-pattern scan passes for the repaired package.
- The archive contains all required code/config files and a tracked documents folder.

## Validation limitation

A full live end-to-end run was not performed in this sandbox because the required third-party packages/model files could not be downloaded due to sandbox network resolution failure. Run the included local commands to validate against your actual PDFs and OpenRouter account.
