use crate::clr::clr_transform;
use crate::loadcsv::load_csv;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;
use smartcore::model_selection::train_test_split;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Fraction of predictions that match the true class label.
fn accuracy(y_true: &[i32], y_pred: &[i32]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

pub fn logisticreg(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let vecunwrap = load_csv(pathfile).unwrap();
    let x_clr = clr_transform(&vecunwrap.0, 1e-6);
    let x = DenseMatrix::from_2d_vec(&x_clr).unwrap();
    // LogisticRegression is a classifier: its label type must implement `Ord`,
    // which `f64` does not (because of NaN). The CSV's target column is loaded
    // as f64, so round it into discrete class labels here.
    let y: Vec<i32> = vecunwrap.1.iter().map(|&v| v.round() as i32).collect();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));
    let model = LogisticRegression::fit(
        &x_train,
        &y_train,
        LogisticRegressionParameters::default().with_alpha(0.8),
    )?;
    let y_hat = model.predict(&x_test)?;
    let acc = accuracy(&y_test, &y_hat);
    println!("=== Model performance (held-out test set) ===");
    println!("Accuracy: {:.4}\n", acc);

    let x_test_vec: Vec<Vec<f64>> = (0..x_test.shape().0)
        .map(|i| (0..x_test.shape().1).map(|j| *x_test.get((i, j))).collect())
        .collect();
    let importances = permutation_logistic(&model, &x_test_vec, &y_test, &vecunwrap.2, 7);
    println!("=== Top 10 taxa by permutation importance ===");
    for (name, score) in importances.iter().take(10) {
        println!("{name:<15} -{score:.4} accuracy when shuffled");
    }
    Ok("Logistic regression has been finished".to_string())
}

pub fn permutation_logistic(
    model: &LogisticRegression<f64, i32, DenseMatrix<f64>, Vec<i32>>,
    x_test: &[Vec<f64>],
    y_test: &[i32],
    taxa_names: &[String],
    seed: u64,
) -> Vec<(String, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_features = x_test[0].len();
    let base_matrix =
        DenseMatrix::from_2d_vec(&x_test.iter().map(|r| r.to_vec()).collect()).unwrap();
    let base_pred = model.predict(&base_matrix).unwrap();
    let base_acc = accuracy(y_test, &base_pred);

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
        let perm_acc = accuracy(y_test, &perm_pred);
        // Positive score = accuracy dropped when this taxon's values were shuffled,
        // i.e. the model relies on it.
        importances.push((taxa_names[j].clone(), base_acc - perm_acc));
    }
    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    importances
}
