use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::mean_squared_error;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn permutation_importance(
    model: &RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>,
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
