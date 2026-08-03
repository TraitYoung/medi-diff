# REST API (v2)

**Stack:** FastAPI · default port **8000** · Swagger at `/docs`.  
Gradio (**7860**) is a **parallel** entry that calls scripts via subprocess — it does **not** call this API.

Chinese field-level tables: [`../developer/API接口文档.md`](../developer/API接口文档.md).

---

## Start

```bash
bash apps/start.sh              # API + Gradio
bash apps/start.sh --api-only
bash apps/stop.sh
# Dev hot-reload for API only:
bash apps/start.sh --api-reload
```

Base URL examples below: `http://localhost:8000`.

| Item | Value |
|------|--------|
| Auth | None (demo / localhost; add your own gateway for exposure) |
| Jobs | Async: `POST` returns `job_id`, poll `GET /jobs/{id}` |

---

## Core endpoints

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Model/data path readiness (+ optional GPU info) |
| `POST` | `/generate/sd15` | Queue SD1.5 + LoRA generation job |
| `POST` | `/review` | Queue evaluation job on a batch |
| `GET` | `/batches` | List generation batches |
| `GET` | `/evaluations` | List eval runs |
| `GET` | `/jobs` · `/jobs/{id}` | Job list / status |
| `GET` | `/reports/latest` · `/history` | Advisor / tuning history |
| `GET` | `/static/*` | Static file serving for outputs |

Exact JSON schemas: open **Swagger** at `/docs` after start.

### Example: health

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Example: poll a job

```bash
curl -s http://localhost:8000/jobs/<job_id> | python3 -m json.tool
```

Prefer the Gradio **Generate** / **One-click pipeline** tabs for interactive use; use the API for automation.

---

## Notes

- Default generation parameters align with `GenParams` / CLI (`strength=0.44`, CFG `7.5`, steps `40`, `--mem-profile` when forwarded).
- Do not expose the unauthenticated API on the public internet without a reverse proxy and auth.
- Errors: follow HTTP status + JSON body; see Chinese API doc §错误码 for the full matrix.
