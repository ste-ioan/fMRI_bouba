# fMRI_bouba

A collection of Python scripts for analyzing functional MRI data using multivariate pattern analysis (MVPA), representational similarity analysis (RSA), and searchlight methods.

## Scripts Overview

- `behavAnalysis.py`: Analyzes behavioral data associated with fMRI experiments. Processes subject responses and performance metrics.

- `mvpa_glmsingle.py`: Performs multivariate pattern analysis on single-trial GLM estimates. Uses machine learning to classify neural patterns.

- `rsa_glmsingle.py`: Implements representational similarity analysis using single-trial GLM data. Compares neural activity patterns across different experimental conditions.

- `searchLight_averaging.py`: Conducts searchlight analysis with signal averaging across voxels to identify informative brain regions.

- `searchLight_mvpa.py`: Implements searchlight with MVPA to map local neural patterns across the brain.

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Scikit-learn
- NiBabel
- Pandas
- Matplotlib/Seaborn

## Usage

Each script is designed to run independently. Modify the input/output paths and parameters directly in the scripts before running.

Example workflow:

1. Analyze behavioral data:
```bash
python behavAnalysis.py
```

2. Run MVPA analysis:
```bash
python mvpa_glmsingle.py
```

3. Compute RSA:
```bash
python rsa_glmsingle.py
```

4. Perform searchlight analyses:
```bash
python searchLight_averaging.py
python searchLight_mvpa.py
```

Note: Make sure to adjust file paths and analysis parameters within each script before running.

## Data Structure

Scripts expect fMRI data in standard neuroimaging formats (NIFTI) and behavioral data in CSV/TSV format. Detailed data requirements are commented at the top of each script.

## Contributing

Feel free to submit issues and enhancement requests.

## Contact

Stefano Ioannucci - [ste.ioannucci@gmail.com](mailto:ste.ioannucci@gmail.com)

## License

[MIT License](LICENSE)
