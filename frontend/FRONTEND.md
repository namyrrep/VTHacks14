# BLACKOUT — Frontend

**Author:** Will Perryman

## Concept

A terminal view displays results and user input. The interactive text box was designed in Figma.

A question returns three answer sentences, one from each of the three best-matching
protocols. Picking one opens that protocol in full, part by part, each part cited to the
page it is printed on and linked into the PDF there.

## Stack

A single `index.html`: markup, styles and script in one file, monospace, dark, blinking
cursor, no framework and no build step. It is served by `backend/server.py`, which also
answers `/api/search`, `/api/protocol` and `/api/stats`, and serves the manual PDFs.

## Layout

1. **Status bar (top).** `OFFLINE · 3 MANUALS · 1,280 PROTOCOLS · 0 EXTERNAL CALLS`. The two
   counts come from `/api/stats`, so they can't drift from what is loaded. "External",
   because the page does call its own machine.
2. **Prompt.** Centered before the first question, pinned to the top after it. Enter submits,
   the input clears, focus returns.
3. **Result area.** One question on screen at a time: a new question replaces the last.

## States

| State | Render |
| --- | --- |
| Answers | Up to 3 choices, each one verbatim sentence, with `[score] MANUAL · PROTOCOL · p.35`. Click, or press 1–3 with the input empty. |
| Protocol | `← BACK TO ANSWERS`, the protocol name, `MANUAL · pp.35–36 · 6 PARTS`, then each part: its page, its subheading, `OPEN PDF →`, and its verbatim text. The part the answer line came from is marked `◂ ANSWER FROM HERE`. |
| Refused | `NOT COVERED IN LOADED PROTOCOLS` + the top score. |
| Backend down | `SEARCH UNAVAILABLE — IS THE SERVER RUNNING?` |

Manuals are cited by short label — Hazmat, Chemical, Medical — which the backend supplies in
`/api/stats` along with each manual's PDF route. Nothing about the manuals is hardcoded here.

The citation shows both page numbers where they differ: `p.172 (printed 158)`. The PDF page
is what the link opens; the printed page is what the paper copy shows. The WV protocols
print no page numbers, so those show the PDF page alone.

## Backend contract

`POST /api/search {question, k:3}` → `{confident, top_score, results[]}`. Each result is one
protocol and carries `answer`, `answer_page`, `answer_chunk_id`, `protocol`, `manual`,
`score`, and `chunk`, whose `passages` array is what the protocol view renders. The full
shape is in [../backend/BACKEND.md](../backend/BACKEND.md).

## Cut if behind

Number-key selection, then the `◂ ANSWER FROM HERE` marker. The three states ship no matter
what — the refusal state is the pitch.
