# Retrieval notes

Implementation rationale for `text.py`, `agent.py` and `evaluate.py`. The measured
headline numbers and the reasoning behind the digest tail are in the README's
"Retrieval quality" section; this file covers the code-level choices.

## Why a hand-rolled BM25

Deliberately not fastembed's `Bm25`. Its stemmer and stopword list are tuned for prose,
while a repository is mostly identifiers, config keys and paths. Splitting those the
way a reader would is worth more here than stemming, and costs no dependency.

`tokens()` yields each whole identifier *and* its camelCase / snake_case / dotted /
hyphenated parts: `allowDelete` also yields `allow` and `delete`, while `zoneclass`
stays whole. Exact literals are precisely what dense vectors are worst at, so they have
to survive intact.

`term_id()` hashes each token to a stable u32 with blake2b. Qdrant sparse indices are
u32, and a hash means there is no vocabulary file to keep in sync with the index.

Document and query vectors are deliberately asymmetric. Qdrant's IDF modifier supplies
the corpus half at query time, so a document stores only the saturated term frequency
and a query stores only its terms. `BM25_AVGDL` is a constant rather than a measured
corpus average — only the ratio to the current document length matters.

## Fusion

Fusion happens in `search_hits()` rather than in Qdrant. A per-collection `FusionQuery`
hands back rank-derived scores, so every collection's best hit ties at `1/(RRF_K+1)` and
merging nine of those is a coin toss. Ranking globally per modality first keeps the
comparison meaningful.

`depth = max(k * 4, 32)` candidates per collection. A depth of 20 measurably loses
recall.

`SPARSE_W = 0.2`: BM25 contributes at a fraction of dense's weight. Measured, not
guessed — at equal weight it drags r@1 from 0.61 to 0.42, because a 20-word question
matches a lot of mid-IDF prose. At 0.2 it adds recall on exact identifiers without that
cost.

## The digest tail

`WIDE = 42`. Hits past `k` come back as one-line digests rather than being dropped.
Measured on `eval.jsonl`: the expected file is in the top 8 for 0.85 of questions but in
the top 50 for 0.96 — the answer is almost always retrieved, the caller just never sees
it. Digests cost about 90 characters each, so showing ranks 9–50 costs roughly 4k
characters and lets the calling agent do its own judging.

`FULL_CHARS` is a budget over full chunk text only. Nothing is ever dropped for it: the
budget decides full-text-versus-digest, not inclusion. The caller is a better relevance
judge than anything local, so it gets the whole pool.

`ask()` passes `wide=0`. Having the *local* model do that judging measurably loses
recall (see the README), so handing it 42 extra paths just burns its context.

## Errors are not "no results"

`_ranked()` skips a collection that 404s — it simply is not built yet. Anything else,
Qdrant being down or a transport error, is collected into `errs` and surfaced as a
warning by `t_search()`. Swallowing those makes an outage read to the caller as a
confident "no results", which is the worst possible failure mode for a retrieval tool.

## Clipping

`clip()` truncates on a line boundary and says so, loudly. A silently chopped file makes
a small model try to *complete* the code instead of answering the question.

## Evaluation

The point of `rag eval` is the comparison between modes, not the absolute numbers. The
`src` field splits hand-written questions from model-generated ones: a question drafted
*from* the answer text reuses its vocabulary, so the synthetic column reads high and is
only comparable with itself.
