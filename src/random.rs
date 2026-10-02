use crate::clr::clr_transform;
use crate::clr::r2_score;
use crate::loadcsv::load_csv;
use crate::permutate::permutation_importance;
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn randomforest(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let vecunwrap = load_csv(pathfile).unwrap();

    // Normalize compositional abundances with CLR before modeling.
    let x_clr = clr_transform(&vecunwrap.0, 1e-6);

    let x = DenseMatrix::from_2d_vec(&x_clr).unwrap();
    let y = vecunwrap.1;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));

    let params = RandomForestRegressorParameters {
        n_trees: 300,
        ..RandomForestRegressorParameters::default().with_seed(42)
    };

    let model = RandomForestRegressor::fit(&x_train, &y_train, params)?;
    let y_hat = model.predict(&x_test)?;

    let mse = mean_squared_error(&y_test, &y_hat);
    let rmse = mse.sqrt();
    let r2 = r2_score(&y_test, &y_hat);

    println!("=== Model performance (held-out test set) ===");
    println!("RMSE: {:.4}", rmse);
    println!("R^2:  {:.4}\n", r2);

    // Recover the plain (non-CLR) test rows for a readable importance report.
    let x_test_vec: Vec<Vec<f64>> = (0..x_test.shape().0)
        .map(|i| (0..x_test.shape().1).map(|j| *x_test.get((i, j))).collect())
        .collect();

    let importances = permutation_importance(&model, &x_test_vec, &y_test, &vecunwrap.2, 7);
    println!("=== Top 10 taxa by permutation importance ===");
    for (name, score) in importances.iter().take(10) {
        println!("{name:<15} +{score:.4} MSE when shuffled");
    }

    Ok("Random forest has been finished".to_string())
}
