# TODOS

Deferred work captured during plan-eng-review (2026-05-28).

---

## TODO-1: Validate last-token pooling vs mean pooling

**What:** Add a `--pooling` argument to `EmotionTracerConfig` (`last` | `mean`). Run `build_vectors.py` with both options on a small emotion set; compare logistic probe balanced accuracy.

**Why:** The current plan uses last-token hidden states as the story-level representation. For decoder-only causal LMs, mean pooling sometimes outperforms last-token for classification. The pipeline's emotion vector quality depends on this choice, but it's never validated.

**Pros:** Confirms (or corrects) a load-bearing design assumption. If mean pooling wins, it's a one-line change at this stage; much harder to retrofit after the tool is in use.

**Cons:** Requires running a real model twice. Only meaningful after the basic pipeline is working end-to-end.

**Context:** The change is in `emotion_vectors.py` and `build_vectors.py` (`collect_last_token_hs` helper that we're adding). Add `pooling: Literal["last", "mean"] = "last"` to `EmotionTracerConfig`. 30-minute experiment once the pipeline is functional.

**Depends on:** Basic pipeline (Tasks 1–9).

---

## TODO-2: Test and harden ':' token detection for Assistant turn markers

**What:** The visualizer marks the ':' token at the start of each Assistant turn with a special indicator. Add a test (or runtime warning) verifying that Mistral's tokenizer actually produces a standalone ':' token in an `[INST] ... [/INST]:` formatted prompt.

**Why:** Mistral's tokenizer behavior depends on the chat template and surrounding whitespace. If ':' is merged with the adjacent token, the marker is silently absent and plot readers see an incomplete visualization with no error.

**Pros:** A silent failure in a visual feature is worse than no feature — it misleads readers. A test or runtime warning makes the feature trustworthy.

**Cons:** Requires tokenizer access (real tokenizer or realistic mock). Minor complexity.

**Context:** At minimum, add a runtime `warnings.warn(...)` in `trajectory.py` or `visualizer.py` if the ':' token is not found in the expected position. A proper test uses `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")` against a known prompt template.

**Depends on:** Task 3 (model_wrapper.py + tokenizer).

---

## TODO-3: Cache completeness check in build_vectors.py

**What:** At startup, `build_vectors.py` should verify all expected `.npz` files exist before reporting "cache hit." If any files are missing (due to a prior crash), recompute them.

**Why:** `build_vectors.py` is a 20-30 minute GPU run. An OOM or keyboard interrupt after saving some but not all emotion vectors leaves the cache directory in a partial state. `trace.py` treats a partial cache as complete and produces silently wrong results.

**Pros:** Prevents silent corruption of results from partial runs. Recoverable with a simple manifest or existence check.

**Cons:** Small added complexity in the cache-check logic. The current "if file exists, skip" pattern needs to become "if all expected files exist, skip."

**Context:** Add a `validate_cache(cache_dir, emotions, methods, layer_indices)` function in `build_vectors.py`. It checks that `vectors_{method}_layer{idx}.npz` exists for each emotion × method × layer combination. If any are missing, print which are missing and recompute only those.

**Depends on:** Task 9 (build_vectors.py), after the multi-layer loop fix (D3) is in.
