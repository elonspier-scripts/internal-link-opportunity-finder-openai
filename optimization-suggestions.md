# Internal Link Tool Optimization Suggestions

## High-Impact Product Improvements

- Add anchor text suggestions per recommendation so implementation is faster.
- Add a weighted priority score (semantic score + page importance + depth).
- Add GSC enrichment (impressions, clicks, average position) to prioritize opportunities with ranking upside.
- Add quick-win filtering (high relevance, not yet linked, high-value destination pages).
- Add optional intent labels (informational, commercial, navigational) and prefer cross-intent links where useful.

## Link Quality and Relevance

- Distinguish existing links by placement (`body`, `nav`, `footer`) so body-link opportunities remain visible when only nav/footer links exist.
- Add exclusion patterns in UI (for example `/tag/`, `/author/`, `/privacy/`, `/cookie/`, `/wp-json/`).
- Add page eligibility checks (noindex, canonical target mismatch, redirects) before recommending links.
- Add cannibalization warnings for very high semantic overlap among pages with similar intent.

## Performance and Scalability

- Keep baseline candidate generation at 50% and perform threshold filtering in the display layer.
- Add configurable "candidate pool per focus URL" (for example top 20) and then apply display `links_per_page`.
- Cache existing-link crawl results with TTL so checks refresh every X hours but remain fast.
- Add batch mode by folder/path for large sites to avoid very large matrix operations.
- Add optional cap on number of focus URLs per run to reduce runtime spikes.

## UX and Workflow

- Add summary counters: `Yes / No / Unknown / Not checked` for existing-link status.
- Add run diagnostics panel: total URLs, focus URLs matched, skipped focus URLs, checked pages, cache hits.
- Add per-row "reason" field (for example: high semantic overlap, same hub, cross-folder opportunity).
- Add grouped export by `Page to Edit (Source)` so implementers can work page-by-page.
- Add a "diff from previous run" export showing only newly discovered opportunities.

## Suggested Next Implementation Steps

1. Add existing-link status summary + diagnostics panel.
2. Add exclusion-pattern controls in sidebar.
3. Add candidate pool + quick-win filter.
4. Add GSC enrichment and weighted priority scoring.
5. Add implementation-first export grouped by source page.
