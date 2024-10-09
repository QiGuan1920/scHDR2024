# scHDR

scHDR is a novel single-cell drug response prediction model that facilitates drug discovery and personalized medicine.

## Usage Example

To demonstrate the functionality of scHDR, let's take the GSE149383 dataset as an example.

### Quick Start

To use scHDR, you can execute the following code in your environment:

```python
from TrainHDR import Train

# Predict drug response for GSE149383 dataset
results_GSE149383, scores_df_GSE149383 = Train(S='GSE149383')
```

The `Train` function takes a dataset identifier (e.g., `GSE149383`) as input and returns the prediction results (`results_GSE149383`) and the evaluation scores (`scores_df_GSE149383`).

## Notebook Example

We also provide a Jupyter Notebook that showcases how to run the scHDR model and provides detailed results for better reproducibility and understanding. T
