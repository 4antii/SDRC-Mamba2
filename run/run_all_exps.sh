cd ..
# LA-2A
python train.py  --config_path ./configs/release/mamba2_mag_mask.yaml --dataset la2a
python train.py  --config_path ./configs/release/mamba2_mag_phase_mask.yaml --dataset la2a
python train.py  --config_path ./configs/release/lstm_raw_32.yaml --dataset la2a
python train.py  --config_path ./configs/release/tcn_100_config.yaml --dataset la2a
python train.py  --config_path ./configs/release/tcn_300_config.yaml --dataset la2a
python train.py  --config_path ./configs/release/s4_c32_f4.yaml --dataset la2a
python train.py  --config_path ./configs/release/s6_16d.yaml --dataset la2a

# # CL-1B
python train.py  --config_path ./configs/release/mamba2_mag_mask.yaml --dataset cl1b
python train.py  --config_path ./configs/release/mamba2_mag_phase_mask.yaml --dataset cl1b
python train.py  --config_path ./configs/release/lstm_raw_32.yaml --dataset cl1b
python train.py  --config_path ./configs/release/tcn_100_config.yaml --dataset cl1b
python train.py  --config_path ./configs/release/tcn_300_config.yaml --dataset cl1b
python train.py  --config_path ./configs/release/s4_c32_f4.yaml --dataset cl1b
python train.py  --config_path ./configs/release/s6_16d.yaml --dataset cl1b

# # Alesis 3630
python train.py  --config_path ./configs/release/mamba2_mag_mask.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/mamba2_mag_phase_mask.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/lstm_raw_32.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/tcn_100_config.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/tcn_300_config.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/s4_c32_f4.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/s6_16d.yaml --dataset alesis3630

#Ablation
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_no_add_losses.yaml --dataset la2a
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ.yaml --dataset la2a
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ_consistency.yaml --dataset la2a

python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_no_add_losses.yaml --dataset cl1b
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ.yaml --dataset cl1b
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ_consistency.yaml --dataset cl1b

python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_no_add_losses.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ.yaml --dataset alesis3630
python train.py  --config_path ./configs/release/ablation/mamba2_mag_phase_phase_circ_consistency.yaml --dataset alesis3630