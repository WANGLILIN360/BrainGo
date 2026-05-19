//! Neuromodulator concentrations & rules (region-level).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Region-level concentrations of the four major modulators (μM).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Serialize, Deserialize)]
pub struct ModulationLevel {
    pub dopamine: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub noradrenaline: f32,
}

/// How each modulator influences neurons in the region.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModulationRule {
    pub dopamine_effect: DopamineEffect,
    pub serotonin_effect: SerotoninEffect,
    pub ach_effect: AchEffect,
    pub ne_effect: NeEffect,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DopamineEffect {
    ModulateSTDP { factor: f32 },
    ModulateThreshold { delta_v: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SerotoninEffect {
    IncreaseThreshold { delta_v_per_um: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AchEffect {
    ModulateRelease { factor_per_um: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NeEffect {
    GlobalGain { factor_per_um: f32 },
}

impl Default for ModulationRule {
    fn default() -> Self {
        Self {
            dopamine_effect: DopamineEffect::ModulateSTDP { factor: 1.0 },
            serotonin_effect: SerotoninEffect::IncreaseThreshold { delta_v_per_um: 0.0 },
            ach_effect: AchEffect::ModulateRelease { factor_per_um: 0.0 },
            ne_effect: NeEffect::GlobalGain { factor_per_um: 0.0 },
        }
    }
}
