[README.md](https://github.com/user-attachments/files/34543977/README.md)
# Detecting the Approach of Sleep from a Single EEG Channel

Code for the continuous analysis in the paper **"Detecting the Approach of Sleep from a Single EEG Channel: Separating Drowsiness from Temporal Drift."**

The study asks whether a single EEG channel can tell how close a person is to falling asleep while they are still scored as awake. This repository contains the analysis that predicts **minutes until sleep onset** as a continuous quantity from the Pz-Oz EEG spectrum.

Because "minutes until onset" is also simply clock time, a model could succeed by tracking slow electrode or skin drift rather than drowsiness. The analysis therefore pairs the real regression with a **same-state placebo**: the identical regression fitted on an earlier stretch of quiet wakefulness, where minutes-to-onset still counts down but the person's state is not changing. Whatever the placebo achieves estimates what the passage of time alone produces.

## Result

Leave-one-subject-out cross-validation on 41 participants (79 recordings usable in both legs) from the Sleep-EDF Expanded Sleep Cassette subset. Spearman correlations between predicted and true minutes-to-onset are computed within each held-out recording and averaged within participant.

## Data

The data are **not** included in this repository. The analysis uses the public [Sleep-EDF Database Expanded](https://doi.org/10.13026/C2X676) from PhysioNet (Sleep Cassette subset, the first 80 recordings in filename order, SC4001E0 through SC4412E0).

## Leakage safeguards

- Scaler, ridge weights and ridge strength are fit on training participants only; ridge strength is chosen by grouped inner cross-validation.
- Train/test splits are made by participant, so both nights of a person are always on the same side.
- Each leg is calibrated only against its own baseline segment.
- Correlations are computed within each recording, never pooled across recordings.

## License

MIT
