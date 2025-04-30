# Boolean + LSI Information Retrieval System

This repository contains a Rust implementation of an Information Retrieval (IR) system combining Boolean retrieval and Latent Semantic Indexing (LSI) to efficiently search and rank documents from a corpus (90min Football Transfer News dataset).

## Overview

- **Boolean Retrieval**: Efficiently retrieves documents matching simple boolean queries (`AND`, `OR`, `NOT`).
- **Latent Semantic Indexing (LSI)**: Enhances Boolean retrieval results by using Singular Value Decomposition (SVD) to capture semantic relationships, improving relevance ranking.

## Prerequisites

- **Rust** (edition 2021)
- **Cargo** (package manager)

### Dependencies

- `csv` for CSV parsing
- `sprs` for sparse matrix representation
- `nalgebra_sparse` for sparse matrix SVD
- `svdlibrs` for efficient Lanczos-based SVD
- `regex` for tokenization
- `ndarray` & `ndarray_linalg` for linear algebra operations

Add these dependencies in your `Cargo.toml`:

```toml
[dependencies]
csv = "1.1"
regex = "1.10"
sprs = "0.11"
nalgebra_sparse = "0.9"
svdlibrs = "0.5"
ndarray = { version = "0.15", features = ["blas"] }
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
```

## Dataset

### Article Files Structure

The dataset consists of individual article files stored in the `articles` directory:

- Each article is stored in a separate text file (article_001.txt, article_002.txt, etc.)
- File format:
  - Line 1: Title
  - Line 2: Date
  - Line 3: Link
  - Line 4+: Content

#### Converting CSV to Individual Files

If you have the original CSV dataset, you can convert it to individual article files using the `csv_to_files` utility:

```sh
cargo run --release --bin csv_to_files
```

This will create an `articles` directory containing all articles as separate text files.

## How to Build

From the root directory of the project, execute:

```sh
cargo build --release
```

## Running Queries

Execute the binary with Cargo:

```sh
cargo run --release --bin nlp
```

### Example

Queries can be modified directly in the source code (`main.rs`). The default query is set to:

- Boolean: `"Sanson AND Morgan"`
- LSI Re-ranking query: `"Morgan Sanson"`

Modify these lines in the `main()` function for different searches:

```rust
let hits = process_query_boolean("your boolean query", &inv, &all);
let q_vec = query_vector("your LSI query", &vocab, &idf, &tok);
```

## Interpreting Results

- **Boolean results** are retrieved without semantic ranking.
- **LSI re-ranking** significantly improves relevance by ranking the Boolean hits based on semantic similarity (cosine similarity).

Output example:

```plaintext
Docs: 6726
Global TF-IDF: 32017×6726, nnz=1163541
Boolean hits: 3
SVD done (100 dims)

LSI re-rank of Boolean hits:

#1   0.854 — Leicester City Considering Move for Fenerbahce Forward Vedat Muriqi
#2   0.801 — Manchester United's Andreas Pereira Flies Out for Lazio Medical Ahead of Loan Move
```

## Performance

Typical build & query times (on Ryzen 5900X):

| Step              | Time (approx.) |
| ----------------- | -------------- |
| File Reading      | ~2s            |
| TF-IDF & indexing | ~2-3s          |
| SVD computation   | ~10-15s        |
| Query projection  | <1ms           |

Subsequent queries execute rapidly due to precomputed SVD.

## Customization Tips

- Adjust `K_LSI` to change the number of latent dimensions.
- Modify `MIN_DF` and `MAX_DF_RATIO` to tune the TF-IDF vocabulary filtering.
- Edit stop words in `STOP` array to influence token filtering.

## License

MIT License
