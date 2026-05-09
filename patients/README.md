# Patient Parameter Fits

## Bruchovsky Data

The Bruchovsky data was taken from a Phase II clinical trial of intermittent therapy [1].

## E-life data

This dataset is taken from an ongoing cancer trial at the Moffitt [3].

## Post-Processing

We subsequently fit a Lotka-Volterra model to this clinical data, using the `patient_fitting.py` script to generate a dataframe of patient profiles (stored in `csv` files). In the final section of the paper, this fitting is based solely off the first treatment cycle - the `truncate_data.py` script was used to generate these shortened clinical records, ensuring compatibility with the fitting functions.

## References

[1] N. Bruchovsky, L. Klotz, J. Crook, S. Malone, C. Ludgate, W. J. Morris, M. E. Gleave and S. L. Goldenberg,
`Final results of the canadian prospective phase II trial of intermittent androgen suppression for men in
biochemical recurrence after radiotherapy for locally advanced prostate cancer', Cancer 107, 389–395
(2006)

[2] M. A. Strobl, J. West, Y. Viossat, M. Damaghi, M. Robertson-Tessi, J. S. Brown, R. A. Gatenby, P. K.
Maini and A. R. Anderson, `Turnover modulates the need for a cost of resistance in adaptive therapy',
Cancer Research 81, 1135–1147 (2020).

[3] J. Zhang, J. J. Cunningham, J. S. Brown and R. A. Gatenby, `Integrating evolutionary dynamics into
treatment of metastatic castrate-resistant prostate cancer', Nature Communications 8, (2017).

[4] J. Zhang, J. J. Cunningham, J. S. Brown and R. A. Gatenby, `Evolution-based mathematical models
significantly prolong response to abiraterone in metastatic castrate-resistant prostate cancer and identify
strategies to further improve outcomes eLife 11:e76284, (2022)
