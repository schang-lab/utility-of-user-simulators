# Pairwise Evaluation Web App

Web interface for the **human pairwise evaluation** of assistant models. Each participant collaborates with two assistant models side-by-side on a writing task: at every turn they send one message, see both replies (randomly labelled "Model A" / "Model B"), pick the one they prefer with a short rationale, and both models continue from the chosen reply. We used this to compare our RL-trained assistant models against baselines with crowdworkers recruited on [Prolific](https://www.prolific.com/).

The study design follows the CollabLLM user study ([Wu et al., 2025](https://arxiv.org/abs/2502.00640), Appendix F). A participant is assigned a document type (blog post, creative writing, or personal statement), chooses a writing intent from a bank, jots down pre-writing notes, then runs a 1-exchange **training session** followed by a ≥5-exchange **real session** on the same topic, and finishes by pasting a ≥200-word final document. The assistants never see the intent or the notes; the participant has to communicate them through the chat. For more details, please refer to the Appendix of our paper.

## Files

| File | Purpose |
|------|---------|
| [app.py](app.py) | FastAPI backend: session/consent flow, writing-task setup, per-turn pairwise generation and preference recording, admin endpoints. Writes one JSON per participant to `results/`. |
| [generation_utils.py](generation_utils.py) | OpenAI-compatible client construction (API providers or a local vLLM server) and parallel two-model generation. |
| [writing_task_bank.py](writing_task_bank.py) | Document types, the 36-intent bank (12 per type), and the pre-writing questions. |
| [static/](static/) | Participant-facing pages, in order: `consent.html` → `instructions.html` → `instructions2.html` → `setup.html` → `task.html` → `done.html` (plus `declined.html` and the shared `study.css`). |
| [run.sh](run.sh) | Launches Uvicorn bound to `127.0.0.1:7860`, meant to sit behind a TLS-terminating reverse proxy. |
| [nginx/eval.conf](nginx/eval.conf) | Reference Nginx site config (HTTP→HTTPS redirect, TLS via certbot, reverse proxy with long timeouts for LLM inference). |
| [quality_check.py](quality_check.py) | Heuristic screen of collected results: on-topic keyword overlap with the assigned intent, AI-text stylometric markers, near-duplicate queries across sessions. |
| [flag_low_quality.py](flag_low_quality.py) | LLM-judge screen for bad-faith submissions. Writes a `<batch>_flags.json` that we use to exclude participants from the win-rate analysis. |

## Installation

```bash
conda create -n usereval python=3.12 -y
conda activate usereval
pip install -r requirements.txt
```

Assistant models are reached through the OpenAI client. Model identifiers are routed by prefix: `gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Gemini, anything else → an OpenAI-compatible server at `http://localhost:<port>/v1` (e.g. vLLM). Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` for whichever API providers you use.

## Running the study

### 1. Serve the assistant models

Start one OpenAI-compatible server per model in the pool. They only need to be reachable from localhost.

```bash
vllm serve /path/to/model-A --served-model-name model-A --port 8001 --host 127.0.0.1
vllm serve /path/to/model-B --served-model-name model-B --port 8002 --host 127.0.0.1
```

### 2. Start the app

```bash
export ADMIN_API_KEY=$(openssl rand -hex 32)     # protects /api/admin/*
export PROLIFIC_COMPLETION_CODE=XXXXXXXX          # shown on the done page
export OPENAI_API_KEY=...                         # optional: enables content moderation of model replies
./run.sh
```

For a quick local check over plain HTTP (no reverse proxy), the browser will reject the `Secure` session cookie unless you also set `COOKIE_SECURE=0`.

### 3. Configure the model pool

The pool is set once through the admin endpoint and persisted to `results/_study_config.json`, so it survives restarts. Each conversation compares two distinct models from the pool; with exactly two entries every participant sees the same pair in a random order.

```bash
curl -X POST http://127.0.0.1:7860/api/admin/study-config \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"models": [{"name": "model-A", "port": 8001},
                  {"name": "model-B", "port": 8002}]}'
```

The `name` must match the model id the server reports under `/v1/models`; the app checks this and rejects the config otherwise. A running tally is available at `GET /api/admin/results` with the same bearer token.

### 4. Send participants to the study

Participants start at `/study`. On Prolific, set the study URL to

```
https://<your-host>/study?PROLIFIC_PID={{%PROLIFIC_PID%}}
```

so the Prolific ID is pre-filled on the consent page. On completion the done page shows the completion code and a "Return to Prolific" button built from `PROLIFIC_COMPLETION_CODE` (or `PROLIFIC_COMPLETION_URL` if set explicitly).

### Results format

Each participant is written to `results/<participant_id>.json` after every state change. The fields that matter for analysis:

```
participant_id, prolific_id, consent_given, completed
writing_setup:   doc_type, intent {id, title, brief}, prewriting_questions, prewriting_answers
conversations[]: session_kind ("training" | "real"), is_training,
                 model1_name, model2_name, setup (snapshot of writing_setup),
                 final_document, finished_at,
                 turns[]: turn_idx, user_query, model1_response, model2_response,
                          model_a_is_model1, preference ("model1" | "model2" | "tie"),
                          preference_ab ("model_a" | "model_b" | "tie"), rationale,
                          query_at, responded_at, preferred_at
reports[], feedback[]
```

`model_a_is_model1` records which model was shown on the left for that turn (the A/B assignment is re-randomised every turn), and `preference` is already mapped back to the underlying model. Training-session conversations carry `is_training: true` and should be excluded from analysis.

## Deployment over HTTPS

Only ports 80/443 need to be public; Uvicorn and the model servers stay on localhost.

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo cp nginx/eval.conf /etc/nginx/sites-available/eval.example.com     # edit the server_name first
sudo ln -s /etc/nginx/sites-available/eval.example.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d eval.example.com
./run.sh
```

If the host has no public IP, a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) pointed at `http://localhost:7860` works in place of Nginx and handles TLS itself.

## Quality control

We screened submissions in two passes before analysis.

```bash
# Heuristic pass: prints a per-participant table and writes results/_quality_check.json
python quality_check.py results/

# LLM-judge pass: flags bad-faith submissions (off-topic / junk queries, meaningless
# rationales, unrelated final document) and writes results_flags.json
export OPENAI_API_KEY=...
python flag_low_quality.py --input-dir results/ --model gpt-5-mini
```

The judge is instructed to default to `ok` and flag only clear disengagement, not poor writing. Participants whose flag is `low_quality` (or who were skipped for having no consent / no real session) were excluded from the win-rate analysis; `borderline` submissions were kept.

## Adapting the study

- **Consent and contact details.** `static/consent.html`, `static/done.html`, and `static/declined.html` contain bracketed placeholders (`[PI name]`, `[contact email]`, `[IRB protocol ID]`, ...) that must be filled in before deploying.
- **Study parameters.** Session lengths and the final-document minimum are `MIN_EXCHANGES_TRAINING`, `MIN_EXCHANGES_REAL`, and `DOC_WORD_MIN` at the top of `app.py`; the client-side minimums (`MIN_QUERY_WORDS`, `MIN_RATIONALE_WORDS` in `task.html`, `MIN_PREWRITING_WORDS` in `setup.html`) and the numbers quoted in the instruction pages should be kept in sync.
- **Writing tasks.** Add or replace intents and pre-writing questions in `writing_task_bank.py`.
- **Evaluation criteria.** The rubric shown to participants above the chat lives in `task.html` (the `rubric-content` block).

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ADMIN_API_KEY` | recommended | Bearer token for `/api/admin/*`. If unset the admin endpoints are open. |
| `PROLIFIC_COMPLETION_CODE` | for Prolific | Completion code shown on the done page; also builds the return URL. |
| `PROLIFIC_COMPLETION_URL` | no | Overrides the auto-built Prolific return URL. |
| `OPENAI_API_KEY` | no | Enables OpenAI moderation of model replies (a flagged reply is replaced by a placeholder; if both are flagged the turn is blocked). Also required for `gpt-*` models and for `flag_low_quality.py`. |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins, only needed if the API is called from another site. |
| `COOKIE_SECURE` | no | Set to `0` for plain-HTTP local testing. Defaults to secure cookies. |
| `PORT` | no | Uvicorn port for `run.sh` (default 7860, must match the Nginx config). |
