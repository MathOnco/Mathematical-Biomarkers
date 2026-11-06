# Mathematical Biomarkers

A mathematical approach to predicting the benefit of Adaptive Therapy

## Lotka - Volterra Model

This is a simple 2-population Lotka-Volterra tumour model, where $S$ is the number of susceptible cells, and $R$ is the number of resistant cells.

$$
\frac{dS}{dt} = r_{S} S \left(1 - \frac{S+R}{K}\right) \times (1-d_{D}D) - d_{S}S, \\\\
\frac{dR}{dt} = r_{R} R \left(1 - \frac{S+R}{K}\right) - d_{R}R
$$

Both species follow a modified logistic growth model with growth rates $r_{S}$ and $r_{R}$, where the total population (rather than the species population) is modified by the carrying capacity $K$.

For the susceptible population, this growth rate is also modified by the drug concentration $D$ and the killing rate of the drug $d_{D}$.

Finally, both species have a natural death rate, of $d_{S}$ and $d_{R}$ respectively.

This model is implemented in the `LotkaVolterraModel` class, which inherits from `ODEModel`. This parent class sets parameters such as error tolerances for the solver, and then solves the ODE model for each treatment period sequentially.
