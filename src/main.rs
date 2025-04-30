//! Boolean filter + LSI rerank for 90min transfer news

use regex::Regex;
use sprs::{CsMat, TriMat};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f64::EPSILON;

use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use nalgebra_sparse::{CooMatrix, CsrMatrix};
use svdlibrs::svd_dim;

/* ------------------------------------------------------------------ */
const ARTICLES_DIR: &str = "articles";
const K_LSI: usize = 100;
const TOP_K: usize = 10;

const MIN_DF: usize = 1;
const MAX_DF_RATIO: f64 = 0.5;
/* ------------------------------------------------------------------ */

#[allow(dead_code)]
#[derive(Debug)]
struct Document {
    title: String,
    date: String,
    link: String,
    content: String,
}

/* ---------------- File reading ----------------------------- */

fn read_articles(dir_path: &str) -> Result<Vec<Document>, Box<dyn Error>> {
    let mut docs = Vec::new();
    let entries = std::fs::read_dir(dir_path)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Skip if not a file
        if !path.is_file() {
            continue;
        }

        // Read file content
        let content = std::fs::read_to_string(&path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Skip if file doesn't have at least 4 lines
        if lines.len() < 4 {
            eprintln!("Skipping file {:?} - not enough lines", path);
            continue;
        }

        // Extract document fields
        let title = lines[0].to_string();
        let date = lines[1].to_string();
        let link = lines[2].to_string();
        let content = lines[3..].join("\n");

        docs.push(Document {
            title,
            date,
            link,
            content,
        });
    }

    Ok(docs)
}

/* ---------------- Boolean engine ----------------- */

fn build_inverted_index(docs: &[Document]) -> HashMap<String, HashSet<usize>> {
    let tok = Regex::new(r"\w+").unwrap();
    let mut idx = HashMap::new();
    for (i, d) in docs.iter().enumerate() {
        for w in tok.find_iter(&d.content.to_lowercase()) {
            idx.entry(w.as_str().to_owned())
                .or_insert_with(HashSet::new)
                .insert(i);
        }
    }
    idx
}

fn process_query_boolean(
    q: &str,
    idx: &HashMap<String, HashSet<usize>>,
    all: &HashSet<usize>,
) -> HashSet<usize> {
    let re_or = Regex::new(r"(?i)\\s+OR\\s+").unwrap();
    let mut res = HashSet::new();
    for clause in re_or.split(q) {
        let mut set: Option<HashSet<usize>> = None;
        let mut neg = false;
        for t in clause.split_whitespace() {
            if t.eq_ignore_ascii_case("AND") {
                continue;
            }
            if t.eq_ignore_ascii_case("NOT") {
                neg = true;
                continue;
            }
            let mut term = idx.get(&t.to_lowercase()).cloned().unwrap_or_default();
            if neg {
                term = all.difference(&term).cloned().collect();
                neg = false;
            }
            set = Some(match set {
                None => term,
                Some(acc) => acc.intersection(&term).cloned().collect(),
            });
        }
        if let Some(cs) = set {
            res = res.union(&cs).cloned().collect();
        }
    }
    res
}

/* -------------- TF-IDF builder ------------------- */

const STOP: &[&str] = &[
    "the", "a", "an", "and", "of", "for", "to", "with", "in", "on", "at", "is", "was", "are", "be",
    "by", "it", "that", "this",
];

fn term_freq(
    text: &str,
    tok: &Regex,
    vocab: &mut HashMap<String, usize>,
    df: &mut Vec<usize>,
) -> HashMap<usize, f64> {
    let mut tf = HashMap::new();
    for m in tok.find_iter(&text.to_lowercase()) {
        let term = m.as_str();
        if STOP.contains(&term) {
            continue;
        }
        let idx = if let Some(&existing) = vocab.get(term) {
            existing
        } else {
            let idx = vocab.len();
            vocab.insert(term.to_owned(), idx);
            df.push(0);
            idx
        };

        *tf.entry(idx).or_insert(0.0) += 1.0;
    }
    for &i in tf.keys() {
        df[i] += 1;
    }
    tf
}

fn build_tfidf_sparse(docs: &[Document]) -> (CsMat<f64>, HashMap<String, usize>, Vec<f64>) {
    let tok = Regex::new(r"\w+").unwrap();
    let mut vocab = HashMap::new();
    let mut df = Vec::<usize>::new();
    let mut doc_tf = Vec::with_capacity(docs.len());

    for d in docs {
        doc_tf.push(term_freq(&d.content, &tok, &mut vocab, &mut df));
    }

    let n_docs = docs.len();
    let mut keep = vec![false; df.len()];
    let mut remap = vec![usize::MAX; df.len()];
    let mut row = 0;
    for (old, &d) in df.iter().enumerate() {
        if d >= MIN_DF && (d as f64) / (n_docs as f64) <= MAX_DF_RATIO {
            keep[old] = true;
            remap[old] = row;
            row += 1;
        }
    }

    let mut idf = vec![0.0; row];
    for (old, &d) in df.iter().enumerate() {
        if keep[old] {
            idf[remap[old]] = ((n_docs as f64) / (d as f64 + EPSILON)).ln();
        }
    }

    let mut tri = TriMat::<f64>::with_capacity((row, n_docs), 0);
    for (j, tf_map) in doc_tf.iter().enumerate() {
        for (&old, &tf) in tf_map {
            if keep[old] {
                let r = remap[old];
                tri.add_triplet(r, j, tf * idf[r]);
            }
        }
    }

    let mut new_vocab = HashMap::new();
    for (term, &old) in &vocab {
        if keep[old] {
            new_vocab.insert(term.clone(), remap[old]);
        }
    }
    (tri.to_csr(), new_vocab, idf)
}

/* ------------- sprs → nalgebra-sparse ------------ */

fn sprs_to_nalgebra(mat: &CsMat<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(mat.rows(), mat.cols());
    for (r, row) in mat.outer_iterator().enumerate() {
        for (c, &v) in row.iter() {
            coo.push(r, c, v);
        }
    }
    CsrMatrix::from(&coo)
}

/* --------------- Lanczos SVD --------------------- */

fn lanczos_svd(mat: &CsMat<f64>, k: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let csr_na = sprs_to_nalgebra(mat);
    let k_eff = k.min(csr_na.nrows().min(csr_na.ncols()).max(2));
    let svd = svd_dim(&csr_na, k_eff).expect("SVD failed");
    let u = svd.ut.t().to_owned();
    let s = svd.s;
    let vt = svd.vt;
    (u, s, vt)
}

/* ---------------- Query utils -------------------- */

fn query_vector(q: &str, vocab: &HashMap<String, usize>, idf: &[f64], tok: &Regex) -> Array1<f64> {
    let mut v = Array1::<f64>::zeros(vocab.len());
    for m in tok.find_iter(&q.to_lowercase()) {
        if let Some(&i) = vocab.get(m.as_str()) {
            v[i] += 1.0;
        }
    }
    for (i, val) in v.iter_mut().enumerate() {
        *val *= idf[i];
    }
    v
}

fn project_query(q: &Array1<f64>, u: &Array2<f64>, s: &Array1<f64>) -> Array1<f64> {
    let mut ql = u.t().dot(q);
    for (i, &sig) in s.iter().enumerate() {
        if sig > EPSILON {
            ql[i] /= sig;
        }
    }
    ql
}

fn project_docs(s: &Array1<f64>, vt: &Array2<f64>) -> Array2<f64> {
    let mut dlat = vt.to_owned();
    for (i, &sig) in s.iter().enumerate() {
        dlat.row_mut(i).mapv_inplace(|v| v * sig);
    }
    dlat
}

/* cosine similarity restricted to hit columns */
fn top_k_cosine_hits(
    q: &Array1<f64>,
    dlat: &Array2<f64>,
    hit_cols: &[usize],
    k: usize,
) -> Vec<(usize, f64)> {
    let qn = q.norm();
    let mut sims: Vec<_> = hit_cols
        .iter()
        .map(|&col| {
            let dv = dlat.column(col); // view of k-vector
            let sim = if qn == 0.0 {
                0.0
            } else {
                q.dot(&dv) / (qn * dv.norm() + EPSILON)
            };
            (col, sim)
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.truncate(k);
    sims
}

/* ------------------------------ main ------------- */

fn main() -> Result<(), Box<dyn Error>> {
    /* 1. read articles from directory */
    let docs = read_articles(ARTICLES_DIR)?;
    println!("Docs: {}", docs.len());

    /* 2. TF-IDF */
    let (csr, vocab, idf) = build_tfidf_sparse(&docs);
    println!(
        "Global TF-IDF: {}×{}, nnz={}",
        csr.rows(),
        csr.cols(),
        csr.nnz()
    );

    /* 3. Boolean filter */
    let inv = build_inverted_index(&docs);
    let all: HashSet<_> = (0..docs.len()).collect();
    let hits = process_query_boolean("Sanson AND Morgan", &inv, &all);
    println!("Boolean hits: {}", hits.len());
    if hits.is_empty() {
        return Ok(());
    }

    /* 4. SVD on full matrix (time is fine) */
    let (u, s, vt) = lanczos_svd(&csr, K_LSI);
    println!("SVD done ({K_LSI} dims)");

    /* 5. Query */
    let tok = Regex::new(r"\w+").unwrap();
    let q_vec = query_vector("Morgan Sanson", &vocab, &idf, &tok);
    let q_lat = project_query(&q_vec, &u, &s);
    let d_lat = project_docs(&s, &vt);

    /* 6. Rank only the hit columns */
    let hit_cols: Vec<usize> = hits.iter().copied().collect();
    println!("\nLSI re-rank of Boolean hits:\n");
    for (rank, (col, sim)) in top_k_cosine_hits(&q_lat, &d_lat, &hit_cols, TOP_K)
        .iter()
        .enumerate()
    {
        println!("#{:<2} {:>6.3} — {}", rank + 1, sim, docs[*col].title);
    }
    Ok(())
}
