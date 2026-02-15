# Biomass HHV Statistical Analysis

This repository contains a reproducible statistical analysis pipeline used in a diploma thesis titled:

**"Primjena modela umjetnih neuronskih mreža u procjeni ogrijevne vrijednosti posliježetvenih ostataka"**

## Description

The script performs:
- Data cleaning and validation
- Descriptive statistics by biomass category
- One-way ANOVA on HHV values
- Tukey HSD post-hoc test with unambiguous interpretation
- Correlation analysis between elemental composition and HHV
- Publication-ready plots (heatmap and boxplot)

Special care is taken to avoid category-order ambiguity by using ordered categorical variables.

## Input data

The dataset must contain the following columns:
- `Kategorija`
- `C`, `H`, `N`, `S`, `O`
- `HHV (Mj kg)`

## Output

- Basic statistics table
- ANOVA table
- Tukey HSD results
- Correlation heatmap
- Boxplot of HHV by biomass category

## Tools and libraries

- Python
- pandas
- numpy
- scipy
- statsmodels
- seaborn
- matplotlib

## Reproducibility

The analysis is fully reproducible and intended for academic and scientific use.

## Author

Diploma thesis work  
Faculty of Agriculture  
University-level academic research
