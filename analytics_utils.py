"""
Walmart Product Analytics - Utility Module
============================================
This module provides reusable classes and functions for cleaning,
transforming, and analyzing the Walmart products dataset.

Demonstrates: OOP (classes, inheritance, encapsulation, polymorphism),
              decorators, generators, type hints, comprehensions,
              error handling, and string formatting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import re
import json
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper functions (demonstrate generators, lambda, closures)
# ---------------------------------------------------------------------------

def price_tier_generator(prices: pd.Series, thresholds: list[float]):
    """Generator that yields (price, tier_label) tuples.

    Demonstrates: generator functions, yield, for-loop, if-elif-else.
    """
    labels = ["Budget", "Mid-Range", "Premium", "Luxury"]
    for price in prices:
        if pd.isna(price):
            yield (price, "Unknown")
        else:
            assigned = False
            for i, threshold in enumerate(thresholds):
                if price <= threshold:
                    yield (price, labels[i])
                    assigned = True
                    break
            if not assigned:
                yield (price, labels[-1])


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a numeric value as currency string.

    Demonstrates: f-string formatting, conditional expression.
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}" if not pd.isna(value) else "N/A"


def safe_json_parse(text: str) -> object:
    """Safely parse a JSON string, returning None on failure.

    Demonstrates: try-except error handling.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DataProcessor:
    """Base class for data processing pipelines.

    Demonstrates: OOP encapsulation, __init__, __repr__, __len__,
                  property decorators, static methods.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe.copy()
        self._original_shape = dataframe.shape
        self._log: list[str] = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rows={len(self._df)}, cols={self._df.shape[1]}, "
            f"operations={len(self._log)})"
        )

    def __len__(self) -> int:
        return len(self._df)

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    @property
    def log(self) -> list[str]:
        return self._log.copy()

    @staticmethod
    def memory_usage_mb(df: pd.DataFrame) -> float:
        """Return DataFrame memory usage in MB."""
        return df.memory_usage(deep=True).sum() / (1024 ** 2)

    def _record(self, message: str):
        self._log.append(message)

    def summary(self) -> dict:
        """Return a summary dict of the current state."""
        return {
            "original_rows": self._original_shape[0],
            "original_cols": self._original_shape[1],
            "current_rows": self._df.shape[0],
            "current_cols": self._df.shape[1],
            "memory_mb": round(self.memory_usage_mb(self._df), 2),
            "operations_performed": len(self._log),
        }


# ---------------------------------------------------------------------------
# Data Cleaner (inherits from DataProcessor)
# ---------------------------------------------------------------------------

class DataCleaner(DataProcessor):
    """Handles all data cleaning and preparation tasks.

    Demonstrates: inheritance, method chaining, while-loop,
                  list/dict comprehensions, string methods.
    """

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)

    # --- Type Conversions ---------------------------------------------------

    def convert_types(self):
        """Cast columns to appropriate data types.

        Demonstrates: dictionary iteration, type casting, try-except.
        """
        # Numeric columns
        numeric_cols = ["final_price", "initial_price", "rating",
                        "review_count", "unit_price"]
        for col in numeric_cols:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

        # Boolean columns
        bool_cols = ["available_for_delivery", "available_for_pickup"]
        bool_map = {"True": True, "true": True, "False": False, "false": False}
        for col in bool_cols:
            if col in self._df.columns:
                self._df[col] = (
                    self._df[col]
                    .astype(str)
                    .map(lambda x: bool_map.get(x, None))
                )

        # Datetime
        if "timestamp" in self._df.columns:
            self._df["timestamp"] = pd.to_datetime(
                self._df["timestamp"], errors="coerce"
            )

        self._record("Converted column data types (numeric, boolean, datetime)")
        return self

    # --- Missing Value Handling ---------------------------------------------

    def handle_missing_values(self):
        """Fill or flag missing values with domain-appropriate defaults.

        Demonstrates: dictionary comprehension, fillna strategies,
                      conditional logic, while-loop for iterative filling.
        """
        # Identify missing percentages
        missing_pct = {
            col: round(self._df[col].isna().mean() * 100, 2)
            for col in self._df.columns
            if self._df[col].isna().any()
        }

        # Fill strategies
        fill_map = {
            "brand": "Unknown",
            "seller": "Unknown",
            "category_name": "Uncategorized",
            "root_category_name": "Uncategorized",
            "free_returns": "No Info",
            "aisle": "Not Specified",
            "unit": "N/A",
            "description": "",
            "colors": "[]",
            "sizes": "[]",
        }

        for col, fill_value in fill_map.items():
            if col in self._df.columns:
                self._df[col] = self._df[col].fillna(fill_value)

        # Numeric: fill initial_price with final_price where missing
        if "initial_price" in self._df.columns and "final_price" in self._df.columns:
            mask = self._df["initial_price"].isna()
            self._df.loc[mask, "initial_price"] = self._df.loc[mask, "final_price"]

        # Rating: fill with median using a while loop approach
        if "rating" in self._df.columns:
            median_rating = self._df["rating"].median()
            idx = 0
            null_indices = self._df.index[self._df["rating"].isna()].tolist()
            while idx < len(null_indices):
                self._df.loc[null_indices[idx], "rating"] = median_rating
                idx += 1

        self._record(
            f"Handled missing values across {len(missing_pct)} columns"
        )
        return self

    # --- Text Cleaning ------------------------------------------------------

    def clean_text_fields(self):
        """Normalize text fields: strip whitespace, fix encoding artifacts.

        Demonstrates: regex, string methods, lambda, apply.
        """
        text_cols = ["product_name", "description", "brand"]
        for col in text_cols:
            if col in self._df.columns:
                self._df[col] = (
                    self._df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(r"&nbsp;|&amp;|&quot;", " ", regex=True)
                )

        self._record("Cleaned text fields (whitespace, encoding artifacts)")
        return self

    # --- Feature Engineering ------------------------------------------------

    def engineer_features(self):
        """Create derived columns for analysis.

        Demonstrates: vectorized operations, np.where, apply with lambda,
                      list comprehension, string slicing, conditional assignment.
        """
        df = self._df

        # Discount amount and percentage
        if "initial_price" in df.columns and "final_price" in df.columns:
            df["discount_amount"] = df["initial_price"] - df["final_price"]
            df["discount_pct"] = np.where(
                df["initial_price"] > 0,
                (df["discount_amount"] / df["initial_price"]) * 100,
                0.0,
            )
            df["discount_pct"] = df["discount_pct"].round(2)
            df["has_discount"] = df["discount_amount"] > 0

        # Price tier using our generator
        if "final_price" in df.columns:
            thresholds = [25.0, 75.0, 200.0]
            tiers = [
                tier for _, tier
                in price_tier_generator(df["final_price"], thresholds)
            ]
            df["price_tier"] = tiers

        # Product name length
        if "product_name" in df.columns:
            df["name_length"] = df["product_name"].apply(len)

        # Number of colors available
        if "colors" in df.columns:
            df["num_colors"] = df["colors"].apply(
                lambda x: len(safe_json_parse(x) or [])
                if isinstance(x, str) and x.startswith("[") else 0
            )

        # Number of sizes available
        if "sizes" in df.columns:
            df["num_sizes"] = df["sizes"].apply(
                lambda x: len(safe_json_parse(x) or [])
                if isinstance(x, str) and x.startswith("[") else 0
            )

        # Rating category
        if "rating" in df.columns:
            conditions = [
                df["rating"] >= 4.5,
                df["rating"] >= 4.0,
                df["rating"] >= 3.0,
                df["rating"] < 3.0,
            ]
            choices = ["Excellent", "Good", "Average", "Below Average"]
            df["rating_category"] = np.select(
                conditions, choices, default="No Rating"
            )

        # Review volume category
        if "review_count" in df.columns:
            df["review_volume"] = pd.cut(
                df["review_count"],
                bins=[-1, 5, 25, 100, float("inf")],
                labels=["Very Low", "Low", "Medium", "High"],
            )

        # Category depth (how many levels in the category path)
        if "categories" in df.columns:
            df["category_depth"] = df["categories"].apply(
                lambda x: len(safe_json_parse(x) or [])
                if isinstance(x, str) else 0
            )

        self._df = df
        self._record("Engineered 10 new features from existing columns")
        return self

    # --- Outlier Detection --------------------------------------------------

    def flag_outliers(self, column: str = "final_price",
                      method: str = "iqr") -> pd.Series:
        """Flag outliers using IQR or Z-score method.

        Demonstrates: if-else branching, numpy operations, polymorphic method.
        """
        series = self._df[column].dropna()

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (self._df[column] < lower) | (self._df[column] > upper)
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            z_scores = (self._df[column] - mean) / std
            outlier_mask = z_scores.abs() > 3
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

        self._df[f"{column}_outlier"] = outlier_mask
        count = outlier_mask.sum()
        self._record(
            f"Flagged {count} outliers in '{column}' using {method.upper()} method"
        )
        return outlier_mask

    # --- Full Pipeline ------------------------------------------------------

    def run_pipeline(self):
        """Execute the full cleaning pipeline via method chaining.

        Demonstrates: method chaining pattern.
        """
        return (
            self.convert_types()
            .handle_missing_values()
            .clean_text_fields()
            .engineer_features()
        )


# ---------------------------------------------------------------------------
# Analytics Engine (inherits from DataProcessor)
# ---------------------------------------------------------------------------

class AnalyticsEngine(DataProcessor):
    """Performs descriptive, diagnostic, and comparative analytics.

    Demonstrates: inheritance, polymorphism, advanced pandas operations,
                  statistical computations, formatted output.
    """

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)

    # --- Descriptive Statistics ---------------------------------------------

    def descriptive_stats(self, column: str) -> dict:
        """Compute comprehensive descriptive statistics for a numeric column.

        Demonstrates: dictionary building, numpy/pandas stats, formatting.
        """
        series = self._df[column].dropna()
        stats = {
            "count": int(series.count()),
            "mean": round(series.mean(), 2),
            "median": round(series.median(), 2),
            "mode": round(series.mode().iloc[0], 2) if len(series.mode()) > 0 else None,
            "std_dev": round(series.std(), 2),
            "variance": round(series.var(), 2),
            "min": round(series.min(), 2),
            "max": round(series.max(), 2),
            "range": round(series.max() - series.min(), 2),
            "q1": round(series.quantile(0.25), 2),
            "q3": round(series.quantile(0.75), 2),
            "iqr": round(series.quantile(0.75) - series.quantile(0.25), 2),
            "skewness": round(series.skew(), 4),
            "kurtosis": round(series.kurtosis(), 4),
            "coeff_of_variation": round((series.std() / series.mean()) * 100, 2)
            if series.mean() != 0 else None,
        }
        self._record(f"Computed descriptive statistics for '{column}'")
        return stats

    # --- Category Analysis --------------------------------------------------

    def category_breakdown(self, group_col: str = "root_category_name",
                           agg_col: str = "final_price") -> pd.DataFrame:
        """Aggregate metrics by category.

        Demonstrates: groupby, agg with named aggregations, sorting.
        """
        result = (
            self._df.groupby(group_col)
            .agg(
                product_count=(agg_col, "count"),
                avg_price=(agg_col, "mean"),
                median_price=(agg_col, "median"),
                min_price=(agg_col, "min"),
                max_price=(agg_col, "max"),
                total_revenue_potential=(agg_col, "sum"),
            )
            .round(2)
            .sort_values("product_count", ascending=False)
        )
        self._record(f"Generated category breakdown by '{group_col}'")
        return result

    # --- Brand Performance --------------------------------------------------

    def brand_performance(self, top_n: int = 15) -> pd.DataFrame:
        """Rank brands by composite performance score.

        Demonstrates: multi-column aggregation, rank, merge, f-string output.
        """
        brand_stats = (
            self._df.groupby("brand")
            .agg(
                num_products=("product_id", "count"),
                avg_price=("final_price", "mean"),
                avg_rating=("rating", "mean"),
                total_reviews=("review_count", "sum"),
                avg_discount_pct=("discount_pct", "mean"),
            )
            .round(2)
        )

        # Composite score: weighted combination of normalized metrics
        for col in ["avg_rating", "total_reviews", "num_products"]:
            col_min = brand_stats[col].min()
            col_max = brand_stats[col].max()
            denom = col_max - col_min if col_max != col_min else 1
            brand_stats[f"{col}_norm"] = (brand_stats[col] - col_min) / denom

        brand_stats["performance_score"] = (
            brand_stats["avg_rating_norm"] * 0.4
            + brand_stats["total_reviews_norm"] * 0.35
            + brand_stats["num_products_norm"] * 0.25
        ).round(4)

        brand_stats = brand_stats.drop(
            columns=[c for c in brand_stats.columns if c.endswith("_norm")]
        )
        brand_stats = brand_stats.sort_values(
            "performance_score", ascending=False
        )

        self._record(f"Computed brand performance rankings (top {top_n})")
        return brand_stats.head(top_n)

    # --- Price vs Rating Correlation ----------------------------------------

    def correlation_analysis(self,
                             columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Compute pairwise correlations for numeric columns.

        Demonstrates: default mutable argument avoidance, select_dtypes.
        """
        if columns is None:
            columns = self._df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self._df[columns].corr().round(4)
        self._record(f"Computed correlation matrix for {len(columns)} columns")
        return corr_matrix

    # --- Discount Effectiveness ---------------------------------------------

    def discount_analysis(self) -> dict:
        """Analyze relationship between discounts and ratings/reviews.

        Demonstrates: boolean indexing, dictionary building, comparison.
        """
        discounted = self._df[self._df["has_discount"]]
        non_discounted = self._df[~self._df["has_discount"]]

        analysis = {
            "discounted_products": len(discounted),
            "non_discounted_products": len(non_discounted),
            "avg_rating_discounted": round(discounted["rating"].mean(), 2),
            "avg_rating_non_discounted": round(non_discounted["rating"].mean(), 2),
            "avg_reviews_discounted": round(discounted["review_count"].mean(), 2),
            "avg_reviews_non_discounted": round(
                non_discounted["review_count"].mean(), 2
            ),
            "avg_discount_percentage": round(discounted["discount_pct"].mean(), 2),
            "median_discount_percentage": round(
                discounted["discount_pct"].median(), 2
            ),
        }
        self._record("Completed discount effectiveness analysis")
        return analysis

    # --- Availability Analysis ----------------------------------------------

    def availability_analysis(self) -> pd.DataFrame:
        """Cross-tabulate delivery and pickup availability by category.

        Demonstrates: crosstab, pivot_table, value_counts.
        """
        availability = (
            self._df.groupby("root_category_name")
            .agg(
                total=("product_id", "count"),
                delivery_available=("available_for_delivery", "sum"),
                pickup_available=("available_for_pickup", "sum"),
            )
        )
        availability["delivery_pct"] = (
            (availability["delivery_available"] / availability["total"]) * 100
        ).round(1)
        availability["pickup_pct"] = (
            (availability["pickup_available"] / availability["total"]) * 100
        ).round(1)

        self._record("Completed availability analysis by category")
        return availability.sort_values("total", ascending=False)

    # --- Text Analytics on Product Names ------------------------------------

    def product_name_insights(self, top_n: int = 20) -> dict:
        """Extract common words and patterns from product names.

        Demonstrates: regex, Counter-like logic, set operations, string ops.
        """
        stop_words = {
            "the", "a", "an", "and", "or", "for", "in", "on", "of", "to",
            "with", "by", "is", "at", "it", "from", "as", "s", "x", "w",
        }

        all_words: list[str] = []
        for name in self._df["product_name"].dropna():
            words = re.findall(r"[a-zA-Z]+", name.lower())
            filtered = [w for w in words if w not in stop_words and len(w) > 2]
            all_words.extend(filtered)

        # Word frequency using dict comprehension
        word_freq: dict[str, int] = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        insights = {
            "total_unique_words": len(word_freq),
            "top_words": sorted_words[:top_n],
            "avg_name_length": round(self._df["name_length"].mean(), 1),
            "longest_name": self._df.loc[
                self._df["name_length"].idxmax(), "product_name"
            ],
            "shortest_name": self._df.loc[
                self._df["name_length"].idxmin(), "product_name"
            ],
        }
        self._record("Extracted product name text insights")
        return insights


# ---------------------------------------------------------------------------
# Report Formatter (demonstrates string formatting, templates)
# ---------------------------------------------------------------------------

class ReportFormatter:
    """Formats analytics results into readable text reports.

    Demonstrates: string formatting (f-strings, .format()),
                  multi-line strings, iteration, conditional display.
    """

    SEPARATOR = "=" * 60
    SUBSEP = "-" * 40

    @staticmethod
    def format_stats_report(stats: dict, title: str = "Descriptive Statistics") -> str:
        lines = [
            ReportFormatter.SEPARATOR,
            f"  {title.upper()}",
            ReportFormatter.SEPARATOR,
        ]
        for key, value in stats.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                lines.append(f"  {label:<25} : {value:>12,.2f}")
            elif isinstance(value, int):
                lines.append(f"  {label:<25} : {value:>12,}")
            else:
                lines.append(f"  {label:<25} : {str(value):>12}")
        lines.append(ReportFormatter.SEPARATOR)
        return "\n".join(lines)

    @staticmethod
    def format_discount_report(analysis: dict) -> str:
        lines = [
            ReportFormatter.SEPARATOR,
            "  DISCOUNT EFFECTIVENESS REPORT",
            ReportFormatter.SEPARATOR,
            "",
            "  {:30} {:>10} {:>10}".format(
                "Metric", "Discounted", "Regular"
            ),
            "  " + ReportFormatter.SUBSEP,
            "  {:30} {:>10,} {:>10,}".format(
                "Product Count",
                analysis["discounted_products"],
                analysis["non_discounted_products"],
            ),
            "  {:30} {:>10.2f} {:>10.2f}".format(
                "Avg Rating",
                analysis["avg_rating_discounted"],
                analysis["avg_rating_non_discounted"],
            ),
            "  {:30} {:>10.1f} {:>10.1f}".format(
                "Avg Review Count",
                analysis["avg_reviews_discounted"],
                analysis["avg_reviews_non_discounted"],
            ),
            "",
            f"  Average Discount: {analysis['avg_discount_percentage']:.1f}%",
            f"  Median Discount:  {analysis['median_discount_percentage']:.1f}%",
            ReportFormatter.SEPARATOR,
        ]
        return "\n".join(lines)

    @staticmethod
    def format_brand_table(brand_df: pd.DataFrame) -> str:
        lines = [
            ReportFormatter.SEPARATOR,
            "  TOP BRAND PERFORMANCE RANKINGS",
            ReportFormatter.SEPARATOR,
            "",
            "  {:<20} {:>6} {:>8} {:>6} {:>8} {:>6}".format(
                "Brand", "Prods", "Avg($)", "Rate", "Reviews", "Score"
            ),
            "  " + "-" * 58,
        ]
        for brand, row in brand_df.iterrows():
            name = str(brand)[:19]
            lines.append(
                "  {:<20} {:>6} {:>8.2f} {:>6.2f} {:>8,} {:>6.4f}".format(
                    name,
                    int(row["num_products"]),
                    row["avg_price"],
                    row["avg_rating"],
                    int(row["total_reviews"]),
                    row["performance_score"],
                )
            )
        lines.append(ReportFormatter.SEPARATOR)
        return "\n".join(lines)
