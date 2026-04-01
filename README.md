# Walmart Product Analytics - End-to-End Data Analysis with Python

A comprehensive data analytics project analyzing 1,000+ Walmart products across multiple categories. This project demonstrates the full data analysis lifecycle — from raw data cleaning to statistical hypothesis testing — using Python's core data science stack.

## Project Overview

This project performs a deep-dive analysis into Walmart's product catalog to uncover pricing patterns, customer sentiment trends, discount effectiveness, and category-level business insights. It is structured as a professional analytics workflow with reusable OOP-based utility modules and a richly visualized Jupyter notebook.

### Objectives

- Clean and prepare a real-world e-commerce dataset with missing values, mixed types, and unstructured fields
- Engineer meaningful features (discount metrics, price tiers, rating categories)
- Perform descriptive, diagnostic, comparative, and text analytics
- Validate findings with statistical hypothesis testing (t-test, Mann-Whitney U, Shapiro-Wilk)
- Deliver actionable business insights with 13+ publication-ready visualizations

## Dataset

| Attribute | Value |
|---|---|
| Source | Walmart.com product listings |
| Records | 1,011 products |
| Features | 44 original + 12 engineered |
| Categories | Home, Clothing, Beauty, Electronics, Food, and more |
| Key Fields | Price, rating, reviews, brand, category, availability, discounts |

## Project Structure

```
Walmart PY/
|-- walmart_analysis.ipynb     # Main analysis notebook (9 sections)
|-- analytics_utils.py         # Reusable OOP utility module
|-- walmart-products.csv       # Raw dataset
|-- plots/                     # Saved visualization outputs
|-- README.md
```

## Technical Skills Demonstrated

### Python Programming

| Concept | Where It's Used |
|---|---|
| **Data Types** | Numeric, string, boolean, datetime, list, dict, tuple, set |
| **String Formatting** | f-strings, `.format()`, alignment operators, currency formatting |
| **Control Flow** | `for` loops, `while` loops, `if-elif-else`, conditional expressions |
| **Comprehensions** | List, dictionary, and generator comprehensions |
| **Functions** | Regular functions, lambda, generators with `yield`, closures |
| **OOP** | Classes, inheritance, encapsulation, `@property`, `@staticmethod`, `__repr__`, `__len__`, method chaining |
| **Error Handling** | `try-except` blocks, `raise` for custom validation |
| **Modules** | Custom module import, organized code separation |

### Data Analysis & Engineering

- **pandas:** `read_csv`, `groupby`, `agg`, `merge`, `apply`, `map`, `cut`, `crosstab`, `pivot_table`, `value_counts`, boolean indexing, method chaining
- **numpy:** Vectorized operations, `np.where`, `np.select`, statistical functions, array broadcasting
- **Feature Engineering:** Discount calculation, price tier classification, rating categorization, review volume binning, text-derived features
- **Data Cleaning:** Type casting, missing value imputation (median fill, forward fill, conditional fill), text normalization, regex-based cleaning
- **Outlier Detection:** IQR method and Z-score method with configurable thresholds

### Statistical Analysis

- Descriptive statistics (mean, median, mode, std, variance, skewness, kurtosis, coefficient of variation)
- Welch's t-test for independent samples
- Mann-Whitney U test (non-parametric)
- Shapiro-Wilk normality test
- Pearson correlation analysis
- Cohen's d effect size calculation

### Data Visualization (13+ Charts)

| Chart Type | Purpose |
|---|---|
| Histogram | Price and rating distributions |
| Box Plot | Rating comparison (discounted vs regular) |
| Violin Plot | Price spread across categories |
| Scatter Plot | Price vs rating with review volume encoding |
| Bubble Chart | Multi-dimensional category comparison |
| Bar Chart | Brand rankings, category counts, word frequencies |
| Horizontal Bar | Top categories, review tags |
| Pie Chart | Price tier proportions |
| Heatmap | Feature correlation matrix |
| Grouped Bar | Delivery vs pickup availability |

## Key Findings

1. **Right-skewed pricing** — Most products fall in the Budget ($0-$25) and Mid-Range ($25-$75) tiers, with luxury outliers above $200
2. **High customer satisfaction** — Average rating exceeds 4.0/5.0, with the majority of products rated "Good" or "Excellent"
3. **Discounts boost engagement** — Discounted products attract significantly more reviews, suggesting price reductions drive customer interaction
4. **Category concentration** — Home and Clothing dominate the catalog, while Beauty products maintain premium pricing with strong ratings
5. **Delivery-first model** — Over 90% of products are available for delivery, while pickup availability varies widely by category
6. **SEO-optimized naming** — Product names are highly descriptive, averaging 70+ characters, packed with searchable keywords

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Execution

1. Clone the repository
2. Open `walmart_analysis.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially — the notebook is self-contained

```bash
jupyter notebook walmart_analysis.ipynb
```

## Resume Description

> **Walmart Product Analytics | Python, pandas, NumPy, Matplotlib, Seaborn, SciPy**
>
> - Built an end-to-end data analytics pipeline analyzing 1,000+ Walmart products across pricing, ratings, availability, and discount effectiveness
> - Designed reusable OOP-based data processing classes with method chaining, inheritance, and encapsulation for scalable data cleaning and feature engineering
> - Engineered 12 derived features including discount metrics, price tiers, and rating categories using pandas vectorized operations and NumPy conditional logic
> - Performed statistical hypothesis testing (Welch's t-test, Mann-Whitney U) to validate discount impact on customer engagement with effect size quantification
> - Created 13+ publication-ready visualizations (heatmaps, violin plots, bubble charts) uncovering pricing patterns, category dynamics, and customer sentiment trends
> - Conducted text analytics on product names and review tags using regex and frequency analysis to identify SEO optimization patterns
