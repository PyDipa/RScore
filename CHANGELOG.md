# Changelog
## [0.1.2] - 2026-06-25

### Changed
- `_fit_single_dist`: la distribuzione Gamma viene ora fittata esclusivamente sui valori positivi (`floc=0`), in accordo con la formulazione SPI canonica (McKee 1993). Il parametro `shift_for_gamma` è stato rimosso.
- `_fit_single_dist`: la frazione di zeri (`qq`) è calcolata sull'intero dataset e salvata nei parametri, eliminando ricalcoli inconsistenti a valle.
- `standardize_data / _pit`: soglia di clipping CDF allineata allo standard SPI (`3.17e-5` ≡ Φ(−4σ)).

### Fixed
- `_fit_single_dist` (KDE): `kde.predictor` → `kde.factor` (AttributeError a runtime).
- `_pit` (Gamma): rimosso `x + shift` non definito (NameError a runtime); `qq` ora letto da `params` anziché ricalcolato sul subset di gruppo.
- `_pit` (Pearson3, KDE): corretta la zero-inflation tramite riscalamento della CDF condizionale su `x > 0`.

### Removed
- Parametro `shift_for_gamma` da `_fit_single_dist` (**breaking change**).
- Chiave `shift_applied` dal dict `params` della Gamma.

### References
- McKee et al. (1993). *J. Am. Meteorol. Soc.*
- Vicente-Serrano et al. (2010). *J. Clim.* 23, 1696–1718.

## [0.1.1] - 2026-06-25
### Fixed
- `standardize_data._pit`: added zero-inflation correction (SPI convention)
  for `gamma`, `pearson3`, and `kde` distributions. CDF is now computed as
  `H(x) = q0 + (1 - q0) * F(x)` before the normal-quantile transform,
  consistent with McKee (1993) and Vicente-Serrano (2010).
- Fixed invalid attribute `kde.predictor` → `kde.factor` in KDE branch of `_pit`.