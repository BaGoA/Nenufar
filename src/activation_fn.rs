/// Activation function trait
/// It allows to define an activation function
pub trait ActivationFunction {
    fn activate(&self, x: f64) -> f64;
}
