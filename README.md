# StressFusion‑RT: Real‑Time Embedded Multi‑Modal Stress Estimation via rPPG, Yawning & Blink Dynamics

## Project Scope

- **Cue 1 (rPPG→HRV):** Extract green-channel pulse from forehead ROI with CHROM/POS; compute HRV (SDNN, RMSSD) as stress biomarkers.
- **Cue 2 (Yawn Detection):** Compute Mouth Aspect Ratio (MAR) from facial landmarks; classify yawns with an SVM or Random Forest on MAR time-series windows.
- **Cue 3 (Blink Irregularity):** Use Eye Aspect Ratio (EAR) to timestamp closures; derive inter-blink interval variance and long-closure counts as fatigue indicators.

## Models and fusion strategies

- **rPPG** → Extract peak intervals → derive HRV features → **Linear Regression** or **Random Forest Regressor** to map to stress score.
- **Yawn** → MAR time-window features → **SVM** (RBF kernel) for binary yawn detection; count yawn rate per minute.
- **Blink** → EAR closures → compute mean/variance of inter-blink intervals → **Logistic Regression** (binarize high vs low irregularity) or feed numeric features into the same regressor as rPPG.    
- **Fusion** → Concatenate all cue-level features into a feature vector → train a **Random Forest** for a continuous stress score.
