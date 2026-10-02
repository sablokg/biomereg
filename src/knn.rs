use crate::clr::clr_transform;
use crate::clr::r2_score;
use crate::loadcsv::load_csv;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::Distances;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::neighbors::knn_regressor::KNNRegressorParameters;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn knnregressor(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let vecunwrap = load_csv(pathfile).unwrap();
    let x_clr = clr_transform(&vecunwrap.0, 1e-6);
    let x = DenseMatrix::from_2d_vec(&x_clr).unwrap();
    let y = vecunwrap.1;
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));
    let model = KNNRegressor::fit(
        &x_train,
        &y_train,
        KNNRegressorParameters::default()
            .with_k(10)
            .with_distance(Distances::euclidian()),
    )?;
    let y_hat = model.predict(&x_test)?;
    let mse = mean_squared_error(&y_test, &y_hat);
    let rmse = mse.sqrt();
    let r2 = r2_score(&y_test, &y_hat);
    println!("=== Model performance (held-out test set) ===");
    println!("RMSE: {:.4}", rmse);
    println!("R^2:  {:.4}\n", r2);

    let x_test_vec: Vec<Vec<f64>> = (0..x_test.shape().0)
        .map(|i| (0..x_test.shape().1).map(|j| *x_test.get((i, j))).collect())
        .collect();
    let importances = permutation_knn(&model, &x_test_vec, &y_test, &vecunwrap.2, 7);
    println!("=== Top 10 taxa by permutation importance ===");
    for (name, score) in importances.iter().take(10) {
        println!("{name:<15} +{score:.4} MSE when shuffled");
    }
    Ok("Random forest has been finished".to_string())
}

pub fn permutation_knn(
    model: &KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>>,
    x_test: &[Vec<f64>],
    y_test: &[f64],
    taxa_names: &[String],
    seed: u64,
) -> Vec<(String, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_features = x_test[0].len();
    let base_matrix =
        DenseMatrix::from_2d_vec(&x_test.iter().map(|r| r.to_vec()).collect()).unwrap();
    let base_pred = model.predict(&base_matrix).unwrap();
    let base_mse = mean_squared_error(&y_test.to_vec(), &base_pred);

    let mut importances = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let mut permuted = x_test.to_vec();
        let mut col: Vec<f64> = permuted.iter().map(|r| r[j]).collect();
        // Fisher-Yates shuffle of this column only
        for i in (1..col.len()).rev() {
            let k = rng.random_range(0..=i);
            col.swap(i, k);
        }
        for (row, &v) in permuted.iter_mut().zip(col.iter()) {
            row[j] = v;
        }
        let perm_matrix = DenseMatrix::from_2d_vec(&permuted).unwrap();
        let perm_pred = model.predict(&perm_matrix).unwrap();
        let perm_mse = mean_squared_error(&y_test.to_vec(), &perm_pred);
        importances.push((taxa_names[j].clone(), perm_mse - base_mse));
    }
    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    importances
}
