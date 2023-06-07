use super::blaf;
use super::num_gen::NumberGenerator;

/// Neural network topology is vector containing the number of neuron for each layers.
/// The first and last layer correspond respectivily to inputs and ouputs.
pub type Topology = Vec<usize>;

/// Neural network is representating by vector of weight matrix and vector of bias vector.
pub struct NeuralNet {
    weigth: Vec<blaf::Matrix>,
    bias: Vec<blaf::ColumnVector>,
}

impl NeuralNet {
    /// Construct a neural network from topology
    pub fn new<Generator>(topology: &Topology, random_gen: &Generator) -> Self
    where
        Generator: NumberGenerator,
    {
        let nb_layer: usize = topology.len();

        let mut weigth: Vec<blaf::Matrix> = Vec::with_capacity(nb_layer);
        let mut bias: Vec<blaf::ColumnVector> = Vec::with_capacity(nb_layer);

        for id in 0..(nb_layer - 1) {
            let nb_rows: usize = topology[id + 1];
            let nb_cols: usize = topology[id];

            weigth.push(blaf::Matrix::new(nb_rows, nb_cols, random_gen));
            bias.push(random_gen.generate_vec(nb_rows));
        }

        return Self { weigth, bias };
    }
}

// Unit test
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
    fn test_neural_net_new() {
        let topology: Topology = vec![2, 3, 1];
        let generator: ZeroGenerator = ZeroGenerator::default();

        let neural_net: NeuralNet = NeuralNet::new(&topology, &generator);

        let size: usize = topology.len() - 1;

        assert_eq!(neural_net.weigth.len(), size);
        assert_eq!(neural_net.bias.len(), size);

        for id in 0..size {
            assert_eq!(neural_net.weigth[id].nb_rows(), topology[id + 1]);
            assert_eq!(neural_net.weigth[id].nb_columns(), topology[id]);
            assert_eq!(neural_net.bias[id].len(), topology[id + 1]);
        }
    }
}
