use super::number_generator::NumberGenerator;

/// Type alias for column-vector
type ColumnVector = Vec<f64>;

/// Row-major matrix representation
struct Matrix {
    nb_rows: usize,
    nb_columns: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Construct a filled zeros row-major matrix
    fn new<Generator>(nb_rows: usize, nb_cols: usize, generator: &Generator) -> Self
    where
        Generator: NumberGenerator,
    {
        return Self {
            nb_rows: nb_rows,
            nb_columns: nb_cols,
            data: generator.generate_vec(nb_rows * nb_cols),
        };
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    // Number generator for tests
    #[derive(Default)]
    struct ZeroGenerator {}

    impl NumberGenerator for ZeroGenerator {
        fn generate_vec(&self, size: usize) -> Vec<f64> {
            return vec![0.0; size];
        }
    }

    #[test]
    fn test_matrix_constructor() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 5;
        let generator: ZeroGenerator = ZeroGenerator::default();

        let row_major_matrix: Matrix = Matrix::new(nb_rows, nb_cols, &generator);
        assert_eq!(row_major_matrix.nb_rows, nb_rows);
        assert_eq!(row_major_matrix.nb_columns, nb_cols);
        assert_eq!(row_major_matrix.data.len(), nb_rows * nb_cols);
    }
}
