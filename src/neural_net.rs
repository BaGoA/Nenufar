use super::activation_fn::ActivationFunction;
use super::blaf;
use super::num_gen::NumberGenerator;
use super::topology::Topology;

/// Neural network is representating by vector of weight matrix, vector of bias vector
/// and vector of activation function
pub struct NeuralNet {
    weigths: Vec<blaf::Matrix>,
    bias: Vec<blaf::ColumnVector>,
    activation_functions: Vec<Box<dyn ActivationFunction>>,
}

impl NeuralNet {
    /// Construct a neural network from topology
    pub fn new<Generator>(topology: Topology, random_gen: &Generator) -> Self
    where
        Generator: NumberGenerator,
    {
        let nb_layer: usize = topology.nb_neurons.len();

        let mut weigths: Vec<blaf::Matrix> = Vec::with_capacity(nb_layer);
        let mut bias: Vec<blaf::ColumnVector> = Vec::with_capacity(nb_layer);

        for id in 0..(nb_layer - 1) {
            let nb_rows: usize = topology.nb_neurons[id + 1];
            let nb_cols: usize = topology.nb_neurons[id];

            weigths.push(blaf::Matrix::new(nb_rows, nb_cols, random_gen));
            bias.push(random_gen.generate_vec(nb_rows));
        }

        return Self {
            weigths,
            bias,
            activation_functions: topology.activation_functions,
        };
    }
}

// Unit test
#[cfg(test)]
mod tests {
    use super::super::topology::TopologyBuilder;
    use super::*;

    // Number generator to fill matrix with zero
    #[derive(Default)]
    struct ZeroGenerator {}

    impl NumberGenerator for ZeroGenerator {
        fn generate_vec(&self, size: usize) -> Vec<f64> {
            return vec![0.0; size];
        }
    }

    // Activation function for test
    #[derive(Default)]
    struct TestActivationFn {}

    impl ActivationFunction for TestActivationFn {
        fn activate(&self, x: f64) -> f64 {
            return x;
        }
    }

    #[test]
    fn test_neural_net_new() {
        let nb_neurons: Vec<usize> = vec![2, 3, 1];

        let topology: Topology = TopologyBuilder::new()
            .nb_input(nb_neurons[0])
            .add_layer(nb_neurons[1], Box::new(TestActivationFn::default()))
            .add_layer(nb_neurons[2], Box::new(TestActivationFn::default()))
            .build()
            .unwrap();

        let generator: ZeroGenerator = ZeroGenerator::default();

        let neural_net: NeuralNet = NeuralNet::new(topology, &generator);

        let size: usize = nb_neurons.len() - 1;

        assert_eq!(neural_net.weigths.len(), size);
        assert_eq!(neural_net.bias.len(), size);
        assert_eq!(neural_net.activation_functions.len(), size);

        for id in 0..size {
            assert_eq!(neural_net.weigths[id].nb_rows(), nb_neurons[id + 1]);
            assert_eq!(neural_net.weigths[id].nb_columns(), nb_neurons[id]);
            assert_eq!(neural_net.bias[id].len(), nb_neurons[id + 1]);
        }
    }
}
