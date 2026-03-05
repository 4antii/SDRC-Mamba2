# Raw
from .tcn.tcn import TCNModel
from .raw.lstm import LSTMModel
from .raw.gru import GRUModel
from .raw.s4_raw import S4Model
from .raw.mamba_raw import MambaRaw

# Magnitude-only model
from .mamba.mamba2_causal_film import Mamba2STFTCausalFilm

# Advanced model
from .advanced.mamba2_mag_phase_mask import Mamba2STFTCausalFilmPhaseMask

# Ablation
from .advanced.mamba2_mag_phase_mask_fix_no_add_losses import Mamba2STFTCausalFilmPhaseMaskFixed as Mamba2STFTCausalFilmPhaseMaskFixedNoAddLosses
from .advanced.mamba2_mag_phase_mask_fix_only_phase_circ import Mamba2STFTCausalFilmPhaseMaskFixed as Mamba2STFTCausalFilmPhaseMaskFixedOnlyPhaseCirc
from .advanced.mamba2_mag_phase_mask_fix_phase_circ_and_consistency import Mamba2STFTCausalFilmPhaseMaskFixed as Mamba2STFTCausalFilmPhaseMaskFixedPhaseCircAndConsistency

models_map = {
    'tcn': TCNModel,
    'lstm': LSTMModel,
    'gru': GRUModel,
    's4_raw': S4Model, 
    'mamba_raw': MambaRaw,
    'mamba2_base_causal_film': Mamba2STFTCausalFilm,
    'mamba2_phase_mask_film': Mamba2STFTCausalFilmPhaseMask,
    # Ablation models
    'mamba2_phase_mask_film_fix_no_add_losses': Mamba2STFTCausalFilmPhaseMaskFixedNoAddLosses,
    'mamba2_phase_mask_film_fix_only_phase_circ': Mamba2STFTCausalFilmPhaseMaskFixedOnlyPhaseCirc,
    'mamba2_mag_phase_mask_fix_phase_circ_and_consistency': Mamba2STFTCausalFilmPhaseMaskFixedPhaseCircAndConsistency
}