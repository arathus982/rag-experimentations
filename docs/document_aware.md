# Document-Aware Chunking Strategy

## Motivation

The three standard chunking strategies (semantic, hierarchical, relational) are designed for long-form documents — technical articles, books, dense reports. The Confluence corpus this system operates on is structurally different:

- **77 documents**, averaging ~19 lines (~250 tokens) each
- **13-level deep folder hierarchy** carrying meaningful domain context
- **Heavy use of markdown tables** that must not be split mid-row
- **Cross-document links** that standard chunkers discard as plain text
- **Emoji-annotated headings** acting as semantic category markers

Running a hierarchical chunker (2048/512/256 tokens) on a 250-token document produces three identical copies of the same chunk at different hierarchy levels. Running a semantic chunker rarely finds a split point, returning the whole document as one chunk anyway. The overhead is real, the benefit is zero.

The document-aware strategy accepts this reality and builds rules that match the actual corpus shape.

---

## Design Rules

### Rule 1 — Small Document Gate

**Threshold:** `DOCUMENT_AWARE_SMALL_DOC_LINE_THRESHOLD = 50 lines`

Documents below this threshold are indexed as a single chunk. No splitting logic runs.

**Why line count, not token count:** Line count is computed with a single `str.count("\n")` — no tokenizer loaded, no model dependency. At the average document size of ~19 lines, the threshold comfortably captures the vast majority of documents as atomic units. This reflects the reality that these are already minimal information units: a Confluence page describing one feature, one permission matrix, one process step.

**Why 50:** Leaves room for documents up to roughly 600 tokens (at ~12 tokens/line average) to be treated as whole. Documents larger than 50 lines are genuinely multi-section and benefit from splitting.

---

### Rule 2 — Heading-Based Section Splitting

For documents that cross the line threshold, the text is split at `##` and `###` boundaries — not at fixed token counts.

**Why headings, not tokens:** Hungarian is agglutinative. A fixed-token boundary can fall mid-suffix and destroy the semantic unit of a word. Heading boundaries are author-defined semantic breaks; they are the only safe split points in this corpus without needing an embedding model to detect meaning shifts (which is what `SemanticSplitterNodeParser` does, at significantly higher cost).

**Code fence guard:** Lines inside fenced code blocks (` ``` `) that happen to start with `##` (e.g. Python comments) are not treated as split points. A boolean `in_code_block` flag is toggled on each ` ``` ` fence line before the heading check runs.

**Table safety:** Table rows always start with `|`, never with `##` or `###`. A heading cannot appear inside a markdown table cell when the cell is a valid table row. The heading prefix check `stripped.startswith("## ")` is already table-safe by construction.

---

### Rule 3 — Folder Path Context Enrichment

Every chunk — whether a whole document or a section — has the following prefix prepended to its text **before embedding**:

```
[Kontextus: <space>/<folder>/<subfolder>]
[Fejezet: <section heading or filename>]

<original text>
```

The folder path is extracted from `doc.metadata["file_path"]` (provided by LlamaIndex's `SimpleDirectoryReader`). Everything between `confluence/` and the filename is used as the path.

**Why:** The 13-level folder hierarchy is the single richest structural signal in this corpus. A question about a feature in `Termékek/Számlázás/Kedvezmények` should surface chunks from that folder above semantically similar-sounding chunks from `Termékek/Számlázás/Díjszabás`. Without path enrichment, both folders' chunks embed to almost the same vector — the product name and billing vocabulary overlap. With enrichment, the folder path shifts the centroid of each chunk's embedding toward its specific domain location.

**Why Hungarian labels (`Kontextus`, `Fejezet`):** The embedding models under evaluation (BGE-M3, Qwen3-Embedding-8B, Harrier-OSS-v1) are all multilingual and handle Hungarian natively. Using Hungarian prefix labels keeps the entire embedded text in one language, avoiding the soft token-weight confusion that English structural labels could introduce for a Hungarian retrieval task.

---

### Rule 4 — Cross-Document Link Extraction

All internal Markdown links — `[text](path)` where `path` does not start with `http` — are extracted from the chunk text and stored in `node.metadata["outbound_links"]` as a list of path strings.

**Why stored and not acted on:** For the current evaluation pass, the links are metadata-only. They establish a ground-truth graph of document relationships derived directly from the authors' own cross-references, without any LLM entity extraction. Future retrieval iterations can use these links for graph expansion: when a retrieved chunk references another document, that document can be fetched and added to the context window. This is a cheaper alternative to `PropertyGraphIndex` for this corpus because the relationship graph is already encoded in the links.

**Scope:** Only internal links are captured. HTTP links (external documentation, external tools) are discarded — they point outside the retrieval corpus and cannot be resolved.

---

## What This Strategy Does Not Do

- **No BM25 / hybrid search differentiation** from other strategies. It uses the same `PGVectorStore` with `hybrid_search=True` as all other strategies. The enrichment prefix is the primary differentiator.
- **No parent-child merge logic.** The small-doc gate eliminates the need for it — there is no meaningful "parent" to merge into when the whole document is already the chunk.
- **No LLM dependency at index time.** Unlike semantic chunking (needs embedding model for split detection) and relational chunking (needs LLM for entity extraction), document-aware chunking runs with zero model calls during document processing. The embedding model is only used when LlamaIndex builds the actual vectors from the already-constructed nodes.

---

## Database Representation

Results are stored with `chunking_strategy = "document_aware"` in the `evaluation_results` table, appearing as a distinct column in `poe report` alongside `semantic`, `hierarchical`, and `relational`.

The PGVectorStore table name follows the same convention: `vectors_{model}_document_aware` (e.g. `vectors_bge_m3_document_aware`).

---

## Expected Behavior on the Current Corpus

| Metric | Expected outcome | Reasoning |
|--------|-----------------|-----------|
| Chunk count | ~90–110 (vs 174 semantic, 742 hierarchical) | Most docs are below 50 lines → one chunk each |
| Context Precision | Higher than hierarchical | No duplicate parent/child copies inflating the pool; folder enrichment improves ranking for domain-specific queries |
| Context Recall | Comparable to semantic | Whole-document chunks preserve all information; section splits preserve section coherence |
| Context Entities Recall | Potentially lower | No explicit entity extraction; depends on whether relevant entities cluster in the same section |
| Index build time | Fastest of all strategies | No model calls during chunking; fewer nodes to embed |
