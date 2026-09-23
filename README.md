# Smart Factory Analytics Platform

Smart factory analytics platform for predictive maintenance, quality analysis, and production optimization using Apache Spark for data processing and TensorFlow for ML models with PowerBI dashboards.

Personal project, built to explore a Spark batch pipeline over factory sensor data. It is not production software — see **Status** below for exactly what is and isn't implemented.

## Status

**Implemented**

- PySpark pipeline reading sensor data and writing aggregates to InfluxDB
- YAML pipeline configuration

**Not implemented / known limitations**

- Pipeline only — no predictive model, quality analyzer or data collector (the earlier README claimed these files; they did not exist)
- Not wired to a real historian or MES
- No tests

## Built with

- **Python** — pyspark, tensorflow, numpy, pandas, influxdb, PyYAML, scikit-learn

## Running it

```bash
pip install -r requirements.txt
python src/spark_pipeline.py
```

## Layout

```
config/
  pipeline.yml
requirements.txt
src/
  spark_pipeline.py
```

