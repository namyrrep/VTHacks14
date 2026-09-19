# Running BLACKOUT in Docker

One image holds the API, the UI, the prebuilt index, the embedding model and the three
manuals. Nothing is fetched at runtime, so the container behaves the same in airplane
mode as the bare-metal app does.

## Demo / submission

```bash
docker compose up --build
```

Then open <http://localhost:8000>. There is no browser inside the container, so nothing
opens on its own — this is the one behavioral difference from `python server.py`.

First start takes ~15 seconds: `server.py` loads the index and warms the model before it
serves anything, exactly as it does natively. Wait for:

```
BLACKOUT is at http://0.0.0.0:8000/   (ctrl-c to stop)
```

Stop with ctrl-c, or `docker compose down`.

A different host port needs no code change — the UI fetches by relative path:

```bash
docker compose up -d && docker compose port blackout 8000   # or edit ports: in compose.yaml
```

## Development

The overlay mounts your working tree over the image's source, so an edit costs a restart
rather than a rebuild.

```bash
# start
docker compose -f compose.yaml -f compose.dev.yaml up

# after editing any .py or index.html
docker compose -f compose.yaml -f compose.dev.yaml restart
```

Restart is ~15 seconds (the model reload). Dependencies and the baked model still come
from the image; only source is overlaid.

Shorten the command by exporting it once per shell:

```bash
export COMPOSE_FILE=compose.yaml:compose.dev.yaml
docker compose up          # now includes the dev overlay
```

## When you actually need a rebuild

| Change | Command | Time |
| --- | --- | --- |
| Python or `index.html` | `restart` (dev) or `docker compose build` | ~15 s |
| Regenerated `backend/data/` | `restart` (dev) or `docker compose build` | ~20 s |
| `backend/requirements-lock.txt` | `docker compose build` | ~8–10 min |
| Nothing cached (fresh machine) | `docker compose build` | ~10 min |

Rebuild and smoke-test once per meaningful feature, not once at the end. The dev overlay
runs your working tree, so a container that works in dev can still fail to build — and
you want to find that out now, not an hour before judging.

## Adding or changing a manual

Index generation stays a native step; the container only consumes the result.

```bash
pip install -r backend/requirements.txt     # native env, if you have not already
cd backend && python ingest.py && python index.py
```

In dev the mount picks up the new `backend/data/` on the next restart. For the sealed
image, `docker compose build`. A new PDF also needs an entry in the `MANUALS` table in
`server.py` to get a short label and a citation link.

## Changing a dependency

`requirements.txt` drives native installs; `requirements-lock.txt` drives the image. Update
both, then rebuild. If anything in the embedding chain moved (torch, transformers,
tokenizers, sentence-transformers, numpy), re-run `python eval_search.py`: those versions
produced the vectors in `index.npy`, and drift shifts scores against `THRESHOLD = 0.45`.

## Troubleshooting

**Port 8000 already in use** — the native server is probably still running. Stop it, or
change the host side of `ports:` in `compose.yaml` to `8001:8000`.

**`OSError` about a missing model at startup** — the baked model layer did not survive.
`HF_HUB_OFFLINE=1` makes this fail loudly by design rather than silently downloading.
Rebuild with `docker compose build --no-cache`.

**PDF citation links 404** — `Documents/` is bundled into the image, so this should not
happen there. In dev it means the bind mount is missing; run from the repo root.

**Build fails resolving `torch==2.14.0+cpu`** — that wheel comes from the extra index URL
at the top of `requirements-lock.txt`; the build needs network access.
