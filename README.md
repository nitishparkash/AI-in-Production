# AI-in-Production — AAVAIL Revenue Prediction Service

End-to-end ML pipeline deployed as a containerized REST API for predicting revenue by country.

## Quick Start

### Local (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Ingest data
python data_ingestion.py

# 2. Run EDA (generates visualizations in eda_outputs/)
python eda.py

# 3. Train models (compares Ridge, RF, GBR — saves best per country)
python model.py

# 4. Start the API
python app.py
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t aavail-api .
docker run -p 8080:8080 aavail-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/countries` | List available countries |
| GET | `/predict/<country>` | Predictions for a specific country |
| GET | `/predict/all` | Predictions for all countries combined |
| GET | `/metrics` | Performance monitoring metrics |
| POST | `/train` | Trigger full model training pipeline |

### Example Requests

```bash
# Health check
curl http://localhost:8080/

# Predict for United Kingdom
curl http://localhost:8080/predict/united_kingdom

# Predict for all countries
curl http://localhost:8080/predict/all

# View metrics
curl http://localhost:8080/metrics

# Trigger training
curl -X POST http://localhost:8080/train
```

## Running Tests

All unit tests can be run with a single command:

```bash
python run_tests.py
```

Test suites:
- **test_api.py** — API endpoint tests (health, predict, train, metrics)
- **test_model.py** — Model training, comparison, prediction, and isolation
- **test_logging.py** — Log file creation, structured events, decorator, isolation

All tests use `TestConfig` which writes to `tests/tmp_*` directories, ensuring complete isolation from production data, models, and logs.

## Project Structure

```
├── app.py                  # Flask REST API
├── model.py                # Model training, comparison & prediction
├── data_ingestion.py       # Data generation, loading & cleaning
├── eda.py                  # Exploratory data analysis with visualizations
├── monitoring.py           # Performance monitoring
├── logger.py               # Centralized structured logging
├── config.py               # Production / Test configuration
├── run_tests.py            # Single-command test runner
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
├── docker-compose.yml      # Container orchestration
├── data/                   # Generated dataset
├── models/                 # Trained model artefacts
├── logs/                   # Application logs
├── eda_outputs/            # EDA visualizations
└── tests/                  # Unit tests (isolated)
```

## Models Compared

| Model | Type |
|-------|------|
| Ridge Regression | **Baseline** — linear model with L2 regularisation |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosted trees |

The best model (by R² on hold-out set) is automatically selected and saved per country.

## Monitoring

The `/metrics` endpoint returns:
- Server uptime
- Per-endpoint call counts and latency stats (avg, min, max)
- System CPU and memory usage
