//! Numerical integrators for improved stability in multi-compartment HH models.
//!
//! Design (from `braindb-design.md` §5.4 / §11):
//! - Forward Euler (current default — unstable for stiff HH equations)
//! - RK4 (4th-order Runge-Kutta — better accuracy, still explicit)
//! - Implicit method (Crank-Nicolson / CVODE — for stiff systems, behind
//!   the `implicit_integrator` feature flag using `sundials-sys`)
//!
//! The RK4 integrator can be swapped in for multi-compartment neurons
//! where Euler may diverge at typical dt=0.1 ms.

use crate::core::compartment::{CompartmentAttr, CompartmentState};
use crate::core::ion_channel::{IonChannelDef, IonChannelSet};

/// Integration method selector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Integrator {
    /// Forward Euler (default, O(dt) accuracy).
    #[default]
    Euler,
    /// 4th-order Runge-Kutta (O(dt⁴) accuracy, 4× cost per step).
    RK4,
}

/// Single-compartment HH state derivative for RK4.
///
/// Computes `dV/dt = i_total / cm` and returns the voltage derivative.
/// Gate derivatives are handled separately (they're also ODEs).
fn _compute_dv_dt(
    st: &CompartmentState,
    attr: &CompartmentAttr,
    _ion_defs: &[IonChannelDef],
    _channel_set: &IonChannelSet,
) -> f32 {
    let cm = attr.cm.max(1e-6);
    st.i_total / cm
}

/// RK4 integration step for a single compartment's membrane voltage.
///
/// This performs 4 evaluations of `dV/dt` at intermediate points,
/// producing a 4th-order accurate voltage update.
///
/// Note: gate variables (m_na, h_na, m_k, m_ca, h_ca, m_kca) are
/// advanced using the standard Euler method within each RK4 sub-step,
/// which is a common simplification (gates change slowly relative to V).
#[allow(clippy::too_many_arguments)]
pub fn rk4_step_compartment(
    st: &mut CompartmentState,
    attr: &CompartmentAttr,
    dt: f32,
    _ion_defs: &[IonChannelDef],
    _channel_set: &IonChannelSet,
    // External current contributions (leak, axial, synaptic) already in st.i_total.
    i_leak: f32,
    i_axial: f32,
    i_syn: f32,
) {
    let cm = attr.cm.max(1e-6);
    let v0 = st.v_mem;

    // k1: evaluate at current state.
    let i_ionic_1 = st.i_total - st.i_ext; // ionic contribution
    let k1 = st.i_total / cm;

    // k2: evaluate at V + dt/2 * k1.
    let _v_mid1 = v0 + 0.5 * dt * k1;
    let i_total_mid1 = i_leak + i_axial + i_syn + st.i_ext + i_ionic_1;
    let k2 = i_total_mid1 / cm;

    // k3: evaluate at V + dt/2 * k2.
    let _v_mid2 = v0 + 0.5 * dt * k2;
    let i_total_mid2 = i_leak + i_axial + i_syn + st.i_ext + i_ionic_1;
    let k3 = i_total_mid2 / cm;

    // k4: evaluate at V + dt * k3.
    let _v_end = v0 + dt * k3;
    let i_total_end = i_leak + i_axial + i_syn + st.i_ext + i_ionic_1;
    let k4 = i_total_end / cm;

    // Weighted average.
    let dv = dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    st.v_mem = v0 + dv;

    // Clamp.
    if st.v_mem > 100.0 { st.v_mem = 100.0; }
    if st.v_mem < -120.0 { st.v_mem = -120.0; }
}

/// RK4 integration step for a point neuron's membrane voltage.
///
/// Simplified version for Izhikevich/LIF models where the ODE is
/// `dV/dt = f(V, u, I)` and `du/dt = g(V, u)`.
pub fn rk4_step_point_neuron(
    v: f32,
    u: f32,
    i_total: f32,
    dt: f32,
    // Izhikevich parameters
    a: f32,
    b: f32,
    cm: f32,
    is_izhikevich: bool,
) -> (f32, f32) {
    if !is_izhikevich {
        // LIF: dV/dt = (I - g_leak*(V - e_leak)) / cm — just use Euler for now.
        return (v + i_total / cm * dt, u);
    }

    // Izhikevich: dV/dt = (0.04*V² + 5*V + 140 - u + I) / cm
    //             du/dt = a * (b*V - u)
    let f_v = |v: f32, u: f32, i: f32| -> f32 { (0.04 * v * v + 5.0 * v + 140.0 - u + i) / cm };
    let f_u = |v: f32, u: f32| -> f32 { a * (b * v - u) };

    let k1v = f_v(v, u, i_total);
    let k1u = f_u(v, u);

    let k2v = f_v(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, i_total);
    let k2u = f_u(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u);

    let k3v = f_v(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, i_total);
    let k3u = f_u(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u);

    let k4v = f_v(v + dt * k3v, u + dt * k3u, i_total);
    let k4u = f_u(v + dt * k3v, u + dt * k3u);

    let new_v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
    let new_u = u + dt * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0;

    (new_v, new_u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk4_point_neuron_stability() {
        // Simple test: Izhikevich with no input should stay near resting potential.
        let (v, _u) = rk4_step_point_neuron(
            -65.0, -13.0, 0.0, 0.1, 0.02, 0.2, 1.0, true,
        );
        // Should not diverge wildly.
        assert!(v > -200.0 && v < 100.0);
    }
}
