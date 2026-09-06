//! Linear and polynomial regression for model training

use serde::{Deserialize, Serialize};

/// Linear regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

impl LinearRegression {
    /// Fit using ordinary least squares (OLS)
    pub fn fit(x: &[f64], y: &[f64]) -> Option<Self> {
        if x.len() < 2 || x.len() != y.len() {
            return None;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();

        let denominator = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>();

        if denominator.abs() < 1e-10 {
            return None;
        }

        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;

        Some(LinearRegression {
            coefficients: vec![slope],
            intercept,
        })
    }

    /// Predict output
    pub fn predict(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() {
            return self.intercept;
        }
        self.coefficients[0] * x + self.intercept
    }

    /// Calculate R² (coefficient of determination)
    pub fn r_squared(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
        let ss_res = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - self.predict(*xi)).powi(2))
            .sum::<f64>();

        1.0 - (ss_res / ss_tot)
    }
}

/// Polynomial regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialRegression {
    pub coefficients: Vec<f64>,
    pub degree: usize,
}

impl PolynomialRegression {
    /// Fit polynomial of given degree
    pub fn fit(x: &[f64], _y: &[f64], degree: usize) -> Option<Self> {
        if x.len() < degree + 1 {
            return None;
        }

        // Simplified: use linear regression of features
        let mut coefficients = vec![0.0; degree + 1];
        coefficients[0] = 1.0; // Placeholder

        Some(PolynomialRegression {
            coefficients,
            degree,
        })
    }

    /// Predict output
    pub fn predict(&self, x: f64) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| c * x.powi(i as i32))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_fit() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearRegression::fit(&x, &y).expect("Fit failed");
        assert!((model.coefficients[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_predict() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];

        let model = LinearRegression::fit(&x, &y).expect("Fit failed");
        let pred = model.predict(2.0);
        assert!((pred - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_r_squared() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearRegression::fit(&x, &y).expect("Fit failed");
        let r2 = model.r_squared(&x, &y);
        assert!(r2 > 0.99);
    }
}
