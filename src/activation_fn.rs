/// Activation function trait
/// It allows to define an activation function and its derivative
pub trait ActivationFunction {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}
