## Summary

Add bulk range extract and fused get+rank APIs on `QWaveletTree`, useful for wavelet-matrix materialization (e.g. contiguous last-column scans in ring/RDF indexes).

### New APIs (inherent on `QWaveletTree`)

| Method | Contract |
|--------|----------|
| `extract_range(range)` | **Ascending multiset** of symbols in the position range (sorted by value with multiplicity). **Not** position order. |
| `extract_range_distinct(range)` / `extract_range_distinct_into` | **Ascending distinct** symbols (one per unique value). |
| `get_and_rank(i)` / `get_and_rank_unchecked` | `(get(i), rank(symbol, i+1))` in a single tree descent. |

### Algorithm

Walk the wavelet matrix level-by-level. Each contiguous range expands into ≤4 child ranges via `rank` at both ends + `occs_smaller`. Digit-order recursion writes sorted output with no intermediate `Vec`s. Singleton ranges short-circuit to a remaining-path get (same pattern as `get_unchecked`).

Algebraically:
- `extract_range(r)` ≡ sort(`get(i)` for `i ∈ r`)
- `extract_range_distinct(r)` ≡ sort+dedup of the same

Empty / out-of-bounds ranges return empty. Works on main-only primitives (no `rank_all` required).

### Tests

- Small fixed sequence + all subranges
- Property tests vs per-row `get` (σ up to 256 and large alphabets)
- `get_and_rank` vs `get` + `rank`
- Empty tree

Also fixes pre-existing `clippy::reversed_empty_ranges` in occs_range tests (quad/bin/huff) so `cargo clippy --lib --tests` is clean of that deny-level lint.

### Out of scope

- Huffman QWT extract (variable-length codes; not digit-order sorted the same way)
- Position-order extract (intentionally not provided — measured slower for the intended use)
- Hybrid expand-depth variants

## Checklist

- [x] `cargo test --lib`
- [x] `cargo clippy --lib --tests` (no new warnings on added code)
