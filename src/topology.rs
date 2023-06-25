use super::activation_fn::ActivationFunction;

/// Neural network topology is define by number of neurons in each layer (including input and
/// output layer), then activation functions apply on each neurons belonging to a layer
pub struct Topology {
    pub layer_dimensions: Vec<usize>,
    pub activation_functions: Vec<Box<dyn ActivationFunction>>,
}

/// Neural network topology builder
pub struct TopologyBuilder {
    nb_input: usize,
    nb_output: usize,
    hidden_layer_dimensions: Vec<usize>,
    activation_functions: Vec<Box<dyn ActivationFunction>>,
}

impl TopologyBuilder {
    /// Construct an empty neural network topology
    pub fn new() -> Self {
        return Self {
            nb_input: 0,
            nb_output: 0,
            hidden_layer_dimensions: Vec::with_capacity(12),
            activation_functions: Vec::with_capacity(10),
        };
    }

    /// Set number of inputs of neural network
    pub fn nb_input(mut self, nb_input: usize) -> Self {
        self.nb_input = nb_input;
        return self;
    }

    /// Set number of outputs of neural network
    fn nb_output(mut self, nb_output: usize) -> Self {
        self.nb_output = nb_output;
        return self;
    }

    /// Add a hidden layer in topology by giving number of neurons and the activation function to apply
    /// on each neuron
    fn add_layer(
        mut self,
        nb_neuron: usize,
        activation_function: Box<dyn ActivationFunction>,
    ) -> Self {
        self.hidden_layer_dimensions.push(nb_neuron);
        self.activation_functions.push(activation_function);
        return self;
    }

    /// Build a neural network topology from pre-configured data
    pub fn build(self) -> Result<Topology, String> {
        if self.nb_input == 0 {
            return Err(String::from(
                "There is no input layer in your neural network",
            ));
        }

        if self.nb_output == 0 {
            return Err(String::from(
                "There is no output layer in your neural network",
            ));
        }

        if self.hidden_layer_dimensions.len() == 0 {
            return Err(String::from(
                "There is no hidden layer in your neural network",
            ));
        }

        let mut layer_dimensions: Vec<usize> =
            Vec::with_capacity(self.hidden_layer_dimensions.len() + 2);

        layer_dimensions.push(self.nb_input);
        layer_dimensions.extend(self.hidden_layer_dimensions.iter());
        layer_dimensions.push(self.nb_output);

        return Ok(Topology {
            layer_dimensions,
            activation_functions: self.activation_functions,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty_topology() {
        let topology: Result<Topology, String> = TopologyBuilder::new().build();
        assert!(topology.is_err());
    }

    struct TestActivationFn {
        factor: f64,
    }

    impl TestActivationFn {
        fn new(factor: f64) -> Self {
            return Self { factor };
        }
    }

    impl ActivationFunction for TestActivationFn {
        fn activate(&self, x: f64) -> f64 {
            if x > 0.0 {
                return self.factor * x;
            } else {
                return 0.0;
            }
        }
    }

    #[test]
    fn test_build_topology_without_inputs() {
        let topology: Result<Topology, String> = TopologyBuilder::new()
            .add_layer(2, Box::new(TestActivationFn::new(2.0)))
            .nb_output(1)
            .build();

        assert!(topology.is_err());
    }

    #[test]
    fn test_build_topology_without_outputs() {
        let topology: Result<Topology, String> = TopologyBuilder::new()
            .add_layer(2, Box::new(TestActivationFn::new(2.0)))
            .nb_input(1)
            .build();

        assert!(topology.is_err());
    }

    #[test]
    fn test_build_topology_without_hidden_layer() {
        let topology: Result<Topology, String> =
            TopologyBuilder::new().nb_input(2).nb_output(1).build();

        assert!(topology.is_err());
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
    fn test_build_topology() {
        let nb_input: usize = 2;
        let nb_output: usize = 1;
        let nb_neuron_first_layer: usize = 4;
        let factor_first_layer: f64 = 3.0;
        let nb_neuron_second_layer: usize = 6;
        let factor_second_layer: f64 = 2.0;

        let topology: Topology = TopologyBuilder::new()
            .nb_input(nb_input)
            .add_layer(
                nb_neuron_first_layer,
                Box::new(TestActivationFn::new(factor_first_layer)),
            )
            .add_layer(
                nb_neuron_second_layer,
                Box::new(TestActivationFn::new(factor_second_layer)),
            )
            .nb_output(nb_output)
            .build()
            .unwrap();

        assert_eq!(topology.layer_dimensions.len(), 4);
        assert_eq!(topology.layer_dimensions[0], nb_input);
        assert_eq!(topology.layer_dimensions[1], nb_neuron_first_layer);
        assert_eq!(topology.layer_dimensions[2], nb_neuron_second_layer);
        assert_eq!(topology.layer_dimensions[3], nb_output);

        assert_eq!(topology.activation_functions.len(), 2);

        let precision: f64 = 0.01;

        assert!(approx_equal(
            topology.activation_functions[0].activate(1.0),
            factor_first_layer,
            precision
        ));

        assert!(approx_equal(
            topology.activation_functions[1].activate(1.0),
            factor_second_layer,
            precision
        ));
    }
}
