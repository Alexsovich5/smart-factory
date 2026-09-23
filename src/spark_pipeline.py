"""
Spark Data Pipeline

Apache Spark pipeline for processing manufacturing sensor data,
aggregating metrics, and feeding ML models.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('spark_pipeline')


class FactoryDataPipeline:
    """Spark-based data processing for factory analytics."""

    def __init__(self, config):
        self.spark = SparkSession.builder \
            .appName("SmartFactoryAnalytics") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
        self.config = config

    def process_sensor_data(self, input_path):
        """Process raw sensor data from manufacturing floor."""
        logger.info("Processing sensor data from %s", input_path)

        df = self.spark.read.json(input_path)

        # Clean and validate
        cleaned = df.filter(
            F.col("temperature").isNotNull() &
            F.col("vibration").isNotNull() &
            F.col("machine_id").isNotNull()
        ).withColumn(
            "timestamp", F.to_timestamp("timestamp")
        )

        # Compute rolling averages
        window = Window.partitionBy("machine_id").orderBy("timestamp").rowsBetween(-10, 0)

        enriched = cleaned.withColumn(
            "avg_temperature", F.avg("temperature").over(window)
        ).withColumn(
            "avg_vibration", F.avg("vibration").over(window)
        ).withColumn(
            "max_vibration", F.max("vibration").over(window)
        ).withColumn(
            "temp_trend", F.col("temperature") - F.col("avg_temperature")
        )

        return enriched

    def aggregate_hourly(self, df):
        """Aggregate sensor data into hourly summaries."""
        return df.groupBy(
            "machine_id",
            F.window("timestamp", "1 hour").alias("time_window")
        ).agg(
            F.avg("temperature").alias("avg_temp"),
            F.max("temperature").alias("max_temp"),
            F.avg("vibration").alias("avg_vibration"),
            F.max("vibration").alias("max_vibration"),
            F.count("*").alias("reading_count"),
            F.stddev("temperature").alias("temp_stddev")
        )

    def detect_anomalies(self, df):
        """Flag anomalous readings using statistical thresholds."""
        stats = df.groupBy("machine_id").agg(
            F.avg("temperature").alias("global_avg_temp"),
            F.stddev("temperature").alias("global_std_temp"),
            F.avg("vibration").alias("global_avg_vib"),
            F.stddev("vibration").alias("global_std_vib")
        )

        joined = df.join(stats, "machine_id")

        return joined.withColumn(
            "is_anomaly",
            (F.abs(F.col("temperature") - F.col("global_avg_temp")) > 3 * F.col("global_std_temp")) |
            (F.abs(F.col("vibration") - F.col("global_avg_vib")) > 3 * F.col("global_std_vib"))
        )

    def compute_oee(self, production_df):
        """Compute Overall Equipment Effectiveness (OEE)."""
        return production_df.groupBy("machine_id", "shift").agg(
            (F.sum("run_time") / F.sum("planned_time")).alias("availability"),
            (F.sum("actual_output") / F.sum("target_output")).alias("performance"),
            (F.sum("good_units") / F.sum("actual_output")).alias("quality")
        ).withColumn(
            "oee", F.col("availability") * F.col("performance") * F.col("quality")
        )

    def write_to_influx(self, df, measurement):
        """Write processed data to InfluxDB."""
        logger.info("Writing to InfluxDB measurement: %s", measurement)
        df.write.format("influxdb").option("measurement", measurement).save()

    def stop(self):
        self.spark.stop()


if __name__ == '__main__':
    import yaml
    with open('config/pipeline.yml') as f:
        config = yaml.safe_load(f)

    pipeline = FactoryDataPipeline(config)
    sensor_data = pipeline.process_sensor_data(config['input_path'])
    hourly = pipeline.aggregate_hourly(sensor_data)
    anomalies = pipeline.detect_anomalies(sensor_data)

    logger.info("Pipeline complete")
    pipeline.stop()
