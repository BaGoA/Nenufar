use super::activation_fn::ActivationFunction;
use super::num_gen::NumberGenerator;

/// Type alias for column-vector
pub type ColumnVector = Vec<f64>;

/// Row-major matrix representation
pub struct Matrix {
    nb_rows: usize,
    nb_columns: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Construct a filled zeros row-major matrix
    pub fn new<Generator>(nb_rows: usize, nb_cols: usize, generator: &Generator) -> Self
    where
        Generator: NumberGenerator,
    {
        return Self {
            nb_rows: nb_rows,
            nb_columns: nb_cols,
            data: generator.generate_vec(nb_rows * nb_cols),
        };
    }

    /// Get number of rows
    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    /// Get number of columns
    pub fn nb_columns(&self) -> usize {
        return self.nb_columns;
    }
}

/// General matrix-vector multiplication
/// This function compute the result of mat*x + y where mat is matrix mxn,
/// x is column vector of n elements and y is column vector of m elements
pub fn gemv(mat: &Matrix, x: &ColumnVector, y: &ColumnVector) -> Result<ColumnVector, String> {
    // Check inputs sizes consistency
    if mat.nb_columns != x.len() {
        return Err(
            "Number of columns of matrix and size of first vector must be equal".to_string(),
        );
    }

    if mat.nb_rows != y.len() {
        return Err("Number of rows of matrix and size of second vector must be equal".to_string());
    }

    // Compute \alpha*mat*x + \beta*y
    let mut vec_res: ColumnVector = vec![0.0; y.len()];

    vec_res.iter_mut().enumerate().for_each(|(index, value)| {
        let slice_lb: usize = index * mat.nb_columns;
        let slice_ub: usize = slice_lb + mat.nb_columns;

        *value = y[index]
            + mat.data[slice_lb..slice_ub]
                .iter()
                .zip(x.iter())
                .map(|(mat_elem, x_elem)| mat_elem * x_elem)
                .sum::<f64>();
    });

    return Ok(vec_res);
}

/// Apply an activation function on each element of column vector
/// Given an activation function f and column vector x = [x1, ..., xn],
/// this function return a column vector y = [f(x1), ..., f(xn)]
pub fn apply_activation_function<Fun>(fun: &Fun, x: &ColumnVector) -> ColumnVector
where
    Fun: ActivationFunction,
{
    return x.iter().map(|&elem| fun.activate(elem)).collect();
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // Number generator to fill matrix with zero
    #[derive(Default)]
    struct ZeroGenerator {}

    impl NumberGenerator for ZeroGenerator {
        fn generate_vec(&self, size: usize) -> Vec<f64> {
            return vec![0.0; size];
        }
    }

    #[test]
    fn test_matrix_new() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 5;
        let generator: ZeroGenerator = ZeroGenerator::default();

        let matrix: Matrix = Matrix::new(nb_rows, nb_cols, &generator);
        assert_eq!(matrix.nb_rows, nb_rows);
        assert_eq!(matrix.nb_columns, nb_cols);
        assert_eq!(matrix.data.len(), nb_rows * nb_cols);
    }

    #[test]
    fn test_gemv_return_error() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 5;
        let generator: ZeroGenerator = ZeroGenerator::default();

        let matrix: Matrix = Matrix::new(nb_rows, nb_cols, &generator);
        let x: ColumnVector = vec![0.0; nb_cols + 1];
        let y: ColumnVector = vec![0.0; nb_rows];

        match gemv(&matrix, &x, &y) {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }

        let u: ColumnVector = vec![0.0; nb_cols];
        let v: ColumnVector = vec![0.0; nb_rows + 1];

        match gemv(&matrix, &u, &v) {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    // Number generator to fill matrix for gemv test
    #[derive(Default)]
    struct GemvGenerator {}

    impl NumberGenerator for GemvGenerator {
        fn generate_vec(&self, _size: usize) -> Vec<f64> {
            return vec![1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 2.0, 2.0, 4.0, 2.0, 1.0];
        }
    }

    // Function to check if two numbers are approximatively equal
    fn approx_equal(value: f64, reference: f64, precision: f64) -> bool {
        let mut error: f64 = (value - reference).abs();

        if reference != 0.0 {
            error /= reference.abs();
        }

        return error < precision;
    }

    #[test]
    fn test_gemv_mat_x_plus_y() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 3;
        let generator: GemvGenerator = GemvGenerator::default();

        let matrix: Matrix = Matrix::new(nb_rows, nb_cols, &generator);
        let x: ColumnVector = vec![3.0, 2.0, 1.0];
        let y: ColumnVector = vec![4.0, 5.0, 2.0, 3.0];

        match gemv(&matrix, &x, &y) {
            Ok(vec_res) => {
                assert_eq!(vec_res.len(), y.len());

                let vec_ref: ColumnVector = vec![14.0, 19.0, 17.0, 20.0];

                for id in 0..y.len() {
                    assert!(approx_equal(vec_res[id], vec_ref[id], 0.01));
                }
            }
            Err(_) => assert!(false),
        }
    }

    #[derive(Default)]
    struct PowerBy {
        exponant: f64,
    }

    impl PowerBy {
        fn new(exponant: f64) -> Self {
            return Self { exponant };
        }
    }

    impl ActivationFunction for PowerBy {
        fn activate(&self, x: f64) -> f64 {
            return x.powf(self.exponant);
        }
    }

    #[test]
    fn test_apply_activation() {
        let exponant: f64 = 2.0;
        let power_by_two: PowerBy = PowerBy::new(exponant);

        let x: ColumnVector = vec![4.0, 5.0, 2.0, 3.0];
        let y: ColumnVector = apply_activation_function(&power_by_two, &x);

        for id in 0..y.len() {
            assert!(approx_equal(y[id], x[id].powf(exponant), 0.01));
        }
    }
}
