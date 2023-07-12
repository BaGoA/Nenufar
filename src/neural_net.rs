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

    /// Predict the output according to input given in argument
    pub fn predict(&self, input: &blaf::ColumnVector) -> Result<blaf::ColumnVector, String> {
        if input.len() != self.weigths[0].nb_columns() {
            return Err(String::from(
                "Number of input are not consistent with topology of neural network",
            ));
        }

        let max_size: usize = self
            .bias
            .iter()
            .max_by_key(|elem| elem.len())
            .unwrap()
            .len();

        let mut output: blaf::ColumnVector = blaf::ColumnVector::with_capacity(max_size);
        output.clone_from(&input);

        for id in 0..self.weigths.len() {
            let neuron_inputs: blaf::ColumnVector =
                blaf::gemv(&self.weigths[id], &output, &self.bias[id])?;

            output.clone_from(&blaf::apply_activation_function(
                &self.activation_functions[id],
                &neuron_inputs,
            ));
        }

        return Ok(output);
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

    // Number generator to fill matrix with one
    #[derive(Default)]
    struct OneGenerator {}

    impl NumberGenerator for OneGenerator {
        fn generate_vec(&self, size: usize) -> Vec<f64> {
            return vec![1.0; size];
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
    fn test_neural_net_predict_with_perceptron() {
        let topology: Topology = TopologyBuilder::new()
            .nb_input(2)
            .add_layer(1, Box::new(TestActivationFn::default()))
            .build()
            .unwrap();

        let generator: OneGenerator = OneGenerator::default();

        let neural_net: NeuralNet = NeuralNet::new(topology, &generator);
        let input: Vec<f64> = vec![1.0, 2.0];

        match neural_net.predict(&input) {
            Ok(output) => {
                assert_eq!(output.len(), 1);
                assert!(approx_equal(
                    output[0],
                    input.iter().sum::<f64>() + 1.0,
                    0.01
                ))
            }
            Err(_) => assert!(false),
        }
    }

    #[test]
    fn test_neural_net_predict_with_simple_topology() {
        let topology: Topology = TopologyBuilder::new()
            .nb_input(2)
            .add_layer(2, Box::new(TestActivationFn::default()))
            .add_layer(1, Box::new(TestActivationFn::default()))
            .build()
            .unwrap();

        let generator: OneGenerator = OneGenerator::default();

        let neural_net: NeuralNet = NeuralNet::new(topology, &generator);
        let input: Vec<f64> = vec![1.0, 2.0];

        match neural_net.predict(&input) {
            Ok(output) => {
                assert_eq!(output.len(), 1);
                assert!(approx_equal(
                    output[0],
                    (input.iter().sum::<f64>() + 1.0) * 2.0 + 1.0,
                    0.01
                ))
            }
            Err(_) => assert!(false),
        }
    }

    #[test]
    fn test_neural_net_predict_should_return_error() {
        let topology: Topology = TopologyBuilder::new()
            .nb_input(2)
            .add_layer(2, Box::new(TestActivationFn::default()))
            .add_layer(1, Box::new(TestActivationFn::default()))
            .build()
            .unwrap();

        let generator: OneGenerator = OneGenerator::default();

        let neural_net: NeuralNet = NeuralNet::new(topology, &generator);
        let input: Vec<f64> = vec![1.0, 2.0, 3.0];

        match neural_net.predict(&input) {
            Ok(_) => assert!(false),
            Err(_) => (),
        }
    }
}
