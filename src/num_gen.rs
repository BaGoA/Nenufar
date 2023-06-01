/// Number generator trait
/// It allows not to depend on specific crate and lets
/// the user free to implement its own generator
pub trait NumberGenerator {
    fn generate_vec(&self, size: usize) -> Vec<f64>;
}
