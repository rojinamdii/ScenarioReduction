# ScenarioReduction
“Seasonal wind power scenario reduction with clustering, representative selection, and statistical evaluation.”
 This study presents a systematic approach for seasonal scenario reduction and representative
 selection in wind power data analysis. Raw active power measurements were first cleaned, with
 missing and zero values replaced by local averages to preserve temporal structure. The data were
 then downsampled and classified into seasonal subsets (Spring, Summer, Autumn, Winter). For
 each season, a three-cluster K-means method identified representative load levelslow, medium,
 and highbased on power output. A single representative scenario from each cluster was selected
 as the point closest to its cluster mean, producing a compact yet descriptive set of seasonal
 scenarios. To evaluate the representativeness of this reduced dataset, descriptive statistics
 (mean, median, variance, range) and distribution-based metrics (Wasserstein distance, Energy
 distance, Continuous Ranked Probability Score, and Total Variation Distance) were computed
 between the full seasonal datasets and their corresponding reduced sets. Results show that the
 reduced scenarios preserve the central tendencies and overall distributional properties of the
 original data with minimal information loss. This methodology provides an effective balance
 between data reduction and statistical fidelity, enabling more efficient modeling and simulation
 of seasonal wind power behavior.
