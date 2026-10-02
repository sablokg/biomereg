use bhtsne::tSNE;
use smartcore::decomposition::pca::PCA;
use smartcore::decomposition::pca::PCAParameters;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/
pub fn pcs_tsne(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);
    let mut datavec: Vec<Vec<f32>> = Vec::new();
    for line in fileread.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        datavec.push(row);
    }
    let n_samples = datavec.len();
    let n_features = datavec[0].len();

    // --- PCA ---
    let data_f64: Vec<Vec<f64>> = datavec
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    let x = DenseMatrix::from_2d_vec(&data_f64)?;
    let pca = PCA::fit(&x, PCAParameters::default().with_n_components(2))?;
    let reduced = pca.transform(&x)?;
    println!("PCA reduced shape: {:?}", reduced.shape());

    // --- bhtsne ---
    // Explicit type annotation is required here: `new()` is on its own
    // statement, separate from the builder chain below, so rustc can't
    // infer the sample type `U` (Vec<f32>) on its own.
    let mut tsne: tSNE<f32, Vec<f32>> = tSNE::new(&datavec);
    tsne.perplexity(25.0).epochs(1000).barnes_hut(
        0.5,
        |sample_a: &Vec<f32>, sample_b: &Vec<f32>| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        },
    );
    let embedding: Vec<f32> = tsne.embedding();

    println!(
        "t-SNE finished: {} samples, {} input features, {} embedding dims",
        n_samples,
        n_features,
        embedding.len() / n_samples.max(1)
    );

    Ok("tsne has finished".to_string())
}
