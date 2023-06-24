use super::activation_fn::ActivationFunction;

// Neural network hidden layer specifications
// We have number of neurons in layer and its activation function
// which apply for each neurons
struct HiddenLayerSpecification {
    nb_neuron: usize,
    activation_function: Box<dyn ActivationFunction>,
}

/// Neural network topology is define by number of neurons in input and output layer,
/// then a vector of hidden layer
pub struct Topology {
    nb_input: usize,
    nb_output: usize,
    hidden_layer_specs: Vec<HiddenLayerSpecification>,
}

/// Neural network topology builder
pub struct TopologyBuilder {
    nb_input: usize,
    nb_output: usize,
    hidden_layer_specs: Vec<HiddenLayerSpecification>,
}

impl TopologyBuilder {
    /// Construct an empty topology
    pub fn new() -> Self {
        return Self {
            nb_input: 0,
            nb_output: 0,
            hidden_layer_specs: Vec::with_capacity(10),
        };
    }

    /// Set number of inputs
    pub fn nb_input(mut self, nb_input: usize) -> Self {
        self.nb_input = nb_input;
        return self;
    }

    /// Set number of outputs
    fn nb_output(mut self, nb_output: usize) -> Self {
        self.nb_output = nb_output;
        return self;
    }

    /// Add a hidden layer in topomogy
    fn add_layer(
        mut self,
        nb_neuron: usize,
        activation_function: Box<dyn ActivationFunction>,
    ) -> Self {
        self.hidden_layer_specs.push(HiddenLayerSpecification {
            nb_neuron,
            activation_function,
        });

        return self;
    }

    /// Build a topology
    pub fn build(self) -> Topology {
        return Topology {
            nb_input: self.nb_input,
            nb_output: self.nb_output,
            hidden_layer_specs: self.hidden_layer_specs,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty_topology() {
        let topology: Topology = TopologyBuilder::new().build();

        assert_eq!(topology.nb_input, 0);
        assert_eq!(topology.nb_output, 0);
        assert_eq!(topology.hidden_layer_specs.len(), 0);
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
            .build();

        assert_eq!(topology.nb_input, nb_input);
        assert_eq!(topology.nb_output, nb_output);
        assert_eq!(topology.hidden_layer_specs.len(), 2);

        assert_eq!(
            topology.hidden_layer_specs[0].nb_neuron,
            nb_neuron_first_layer
        );

        assert_eq!(
            topology.hidden_layer_specs[1].nb_neuron,
            nb_neuron_second_layer
        );

        let precision: f64 = 0.01;

        assert!(approx_equal(
            topology.hidden_layer_specs[0]
                .activation_function
                .activate(1.0),
            factor_first_layer,
            precision
        ));

        assert!(approx_equal(
            topology.hidden_layer_specs[1]
                .activation_function
                .activate(1.0),
            factor_second_layer,
            precision
        ));
    }
}
