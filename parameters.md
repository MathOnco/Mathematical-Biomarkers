# Parameters

This contains a list of parameters used in the model, alongside their default values and meaning.

## Tumour model parameters

- `n0` = 0.75 - _(Initial size of tumour)_
- `rFrac` = 0.001 - _(Initial proportion of susceptible cells)_
- `rS` = 0.027 - _(Birth rate of susceptible cells)_
- `rR` = 0.027 - _(Birth rate of resistant cells)_
- `dS` = 0.0 - _(Death rate of susceptible cells)_
- `dR` = 0.0 - _(Death rate of resistant cells)_
- `dD` = 1.5 - _(Drug induced death rate of susceptible cells)_
- `K` = 1.0 - _(Carrying capacity of tumour environment)_
- `D` = 0 - _(Drug concentration in environment)_
- `theta` = 1 - _(Scale factor for mapping cell counts to observed fluorescent area - assumes a cell radius of 10uM)_
- `DMax` = 1.0 - _(Max doseage given to patient)_
- `S0` = n0 * (1 - rFrac) - _(Initial size of susceptible portion of tumour)_
- `R0` = (n0 * rFrac) - _(Initial number of resistant portion of tumour)_
