# BLACKOUT — Frontend Implementation Plan

**Author:** Will Perryman

## Concept

A terminal view displays results and user input. The interactive text box was designed in Figma.

## Stack

Single `index.html` + `style.css` + `app.js`. Monospace font, dark background, blinking cursor. No framework.

## Layout

1. **Status bar (top).** `AIRPLANE MODE · 3 MANUALS · 412 SECTIONS · 0 NETWORK CALLS`
2. **Scrollback (middle).** Query history and results, newest at bottom, auto-scrolled.
3. **Prompt (bottom).** The Figma text box. Enter submits, input clears, focus returns.

## Result states

| State | Render |
| --- | --- |
| Confident | Passage verbatim, then `[MANUAL · SECTION · p.172 · 0.81]` |
| Ambiguous | Clarifying question + best general guide |
| Refused | `NOT COVERED IN LOADED PROTOCOLS` + top score |

## Backend contract

Calls `search(question, k=3)` → `{confident, top_score, results[{text, manual, section, page, score}]}`. Build against a hardcoded `fake_search()` first; swap in the real one at integration.

## Build order

1. Static shell + status bar
2. Prompt wired to fake data
3. Result cards, all three states
4. Figma styling pass
5. Swap in real search

## Cut if behind

Styling pass, then scrollback history. The three result states ship no matter what — the refusal state is the pitch.