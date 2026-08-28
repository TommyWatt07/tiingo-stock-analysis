# Tiingo Energy Stock Analysis

Research project analysing relationships between energy stocks using
historical price data.

## Analysis

The project investigates:

- Pairwise stock correlations
- Correlation matrices
- K-means clustering
- Recent vs historical cluster stability
- PCA
- Rolling correlations
- Autocorrelation
- Synchronous autocorrelation regimes
- Autocorrelation episodes
- Stock replication using correlation/LASSO
- Stock replication using PCA

## Data

Historical stock price data is obtained from Tiingo.

## Methodology

See the research report:

`report/energy_stock_analysis_report.docx`

## Project Structure

- `src/` - analysis code
- `results/` - generated results and visualisations
- `report/` - written research report

## Running the Analysis

```powershell
python -m src.main