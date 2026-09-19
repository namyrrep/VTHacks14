# BLACKOUT — Offline Field Protocol Assistant

Answers plain-English questions from federal and state emergency response guides, with no network connection of any kind.

Built at VTHacks 14 for the Peraton Mission-Critical AI track.

## Authors

- William Perryman
- Edwin Barrack

## Why

When responders most need a protocol manual, the network is often gone — grid outages outlast cell-site batteries, backhaul gets cut, networks congest during mass-casualty events, and basements and tunnels have no signal at all. BLACKOUT has no dependency to lose.

It also never generates text. Every answer is the manual's own wording, shown with its source and page number, so it can be verified. When a question isn't covered by the loaded protocols, it says so instead of guessing.

## Features

- **Fully offline.** No API keys, no server, no network calls. Runs in airplane mode.
- **Cited answers.** Every result shows the manual, section, and page.
- **Abstains when unsure.** Below the confidence threshold it refuses rather than guessing.
- **Semantic search.** Matches meaning, not keywords — "can't catch their breath" finds "respiratory distress."
- **Possible online features (future).** Store-and-forward sync for field reports, protocol updates when a link is available.

## Sources

| Manual | Domain |
| --- | --- |
| ERG 2024 (PHMSA) | Hazmat and transport incidents |
| NIOSH Pocket Guide to Chemical Hazards | Chemical exposure, symptoms, first aid |
| West Virginia First Responder Guide | Emergency medical |

## Implementation

TBD

<!-- Fill in: PyMuPDF for PDF extraction, sentence-transformers (all-MiniLM-L6-v2)
     for local embeddings, numpy cosine similarity for ranking, Streamlit for the UI.
     Note the index size once you have it: N sections across 3 manuals. -->

## Running the application

TBD

<!-- Fill in once it runs:
     pip install -r requirements.txt
     streamlit run app.py
     Note that the index is committed, so no build step is needed. -->

## License

Protocol documents are U.S. government public domain works.