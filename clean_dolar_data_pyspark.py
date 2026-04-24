from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


EXPECTED_COLUMNS = {
    "dia": T.IntegerType(),
    "inflacion": T.DoubleType(),
    "tasa_interes": T.DoubleType(),
    "precio_dolar": T.DoubleType(),
}

CANONICAL_ALIASES = {
    "dia": "dia",
    "day": "dia",
    "inflacion": "inflacion",
    "inflation": "inflacion",
    "tasa_interes": "tasa_interes",
    "tasa_de_interes": "tasa_interes",
    "interest_rate": "tasa_interes",
    "precio_dolar": "precio_dolar",
    "precio_del_dolar": "precio_dolar",
    "dollar_price": "precio_dolar",
}


def normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def build_column_mapping(columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    taken = set()
    for original in columns:
        normalized = normalize_column_name(original)
        canonical = CANONICAL_ALIASES.get(normalized, normalized)
        if canonical in taken:
            continue
        mapping[original] = canonical
        taken.add(canonical)
    return mapping


def cast_and_validate(df: DataFrame) -> Tuple[DataFrame, int, int]:
    df_casted = df.select(
        F.col("dia").cast("int").alias("dia"),
        F.regexp_replace(F.col("inflacion"), ",", ".").cast("double").alias("inflacion"),
        F.regexp_replace(F.col("tasa_interes"), ",", ".").cast("double").alias("tasa_interes"),
        F.regexp_replace(F.col("precio_dolar"), ",", ".").cast("double").alias("precio_dolar"),
    )

    invalid_condition = (
        F.col("dia").isNull()
        | F.col("inflacion").isNull()
        | F.col("tasa_interes").isNull()
        | F.col("precio_dolar").isNull()
        | (F.col("dia") <= 0)
    )
    invalid_rows = df_casted.filter(invalid_condition).count()
    valid_df = df_casted.filter(~invalid_condition)
    return valid_df, invalid_rows, df_casted.count()


def deduplicate_by_day(df: DataFrame) -> Tuple[DataFrame, int]:
    with_id = df.withColumn("_ingest_id", F.monotonically_increasing_id())
    window = Window.partitionBy("dia").orderBy(F.col("_ingest_id"))
    deduped = (
        with_id.withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_ingest_id")
    )
    removed = with_id.count() - deduped.count()
    return deduped, removed


def add_missing_days(df: DataFrame) -> Tuple[DataFrame, int, int, int]:
    bounds = df.agg(F.min("dia").alias("min_dia"), F.max("dia").alias("max_dia")).first()
    min_dia = int(bounds["min_dia"])
    max_dia = int(bounds["max_dia"])

    all_days = (
        df.sparkSession.range(min_dia, max_dia + 1)
        .select(F.col("id").cast("int").alias("dia"))
    )

    joined = all_days.join(df, on="dia", how="left")
    inserted_flag = (
        F.col("inflacion").isNull() & F.col("tasa_interes").isNull() & F.col("precio_dolar").isNull()
    )
    with_flag = joined.withColumn("dia_insertado", inserted_flag.cast("int"))
    inserted_count = with_flag.filter(F.col("dia_insertado") == 1).count()
    return with_flag, inserted_count, min_dia, max_dia


def count_nulls(df: DataFrame, cols: List[str]) -> Dict[str, int]:
    row = df.select(
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    ).first()
    return {c: int(row[c]) for c in cols}


def percentile_pair(df: DataFrame, col_name: str) -> Tuple[float, float]:
    q = df.select(
        F.percentile_approx(F.col(col_name), [0.25, 0.75], 10000).alias("q")
    ).first()["q"]
    return float(q[0]), float(q[1])


def percentile_single(df: DataFrame, col_name: str, p: float) -> float:
    return float(
        df.select(F.percentile_approx(F.col(col_name), p, 10000).alias("p")).first()["p"]
    )


def impute_column(df: DataFrame, col_name: str, order_col: str = "dia") -> Tuple[DataFrame, int]:
    median_value = percentile_single(df.filter(F.col(col_name).isNotNull()), col_name, 0.5)
    # Explicit partition avoids noisy warnings for unpartitioned window execution.
    w_prev = Window.partitionBy(F.lit(1)).orderBy(order_col).rowsBetween(Window.unboundedPreceding, 0)
    w_next = Window.partitionBy(F.lit(1)).orderBy(order_col).rowsBetween(0, Window.unboundedFollowing)

    prev_col = f"_{col_name}_prev"
    next_col = f"_{col_name}_next"
    flag_col = f"imputado_{col_name}"

    enriched = (
        df.withColumn(prev_col, F.last(F.col(col_name), ignorenulls=True).over(w_prev))
        .withColumn(next_col, F.first(F.col(col_name), ignorenulls=True).over(w_next))
        .withColumn(flag_col, F.col(col_name).isNull().cast("int"))
    )

    imputed = (
        enriched.withColumn(
            col_name,
            F.when(
                F.col(col_name).isNull(),
                F.when(
                    F.col(prev_col).isNotNull() & F.col(next_col).isNotNull(),
                    (F.col(prev_col) + F.col(next_col)) / F.lit(2.0),
                ).otherwise(F.coalesce(F.col(prev_col), F.col(next_col), F.lit(median_value))),
            ).otherwise(F.col(col_name)),
        )
        .drop(prev_col, next_col)
    )
    imputed_count = imputed.filter(F.col(flag_col) == 1).count()
    return imputed, imputed_count


def winsorize_iqr(df: DataFrame, col_name: str) -> Tuple[DataFrame, int, float, float]:
    q1, q3 = percentile_pair(df.filter(F.col(col_name).isNotNull()), col_name)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    flag_col = f"outlier_{col_name}"

    flagged = df.withColumn(
        flag_col, ((F.col(col_name) < F.lit(low)) | (F.col(col_name) > F.lit(high))).cast("int")
    )
    outlier_count = flagged.filter(F.col(flag_col) == 1).count()
    capped = flagged.withColumn(
        col_name,
        F.when(F.col(col_name) < F.lit(low), F.lit(low))
        .when(F.col(col_name) > F.lit(high), F.lit(high))
        .otherwise(F.col(col_name)),
    )
    return capped, outlier_count, low, high


def run_pipeline(input_path: str, output_dir: str, report_path: str) -> None:
    spark = (
        SparkSession.builder.appName("LimpiezaProfundaDolar")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(input_path)
    )
    raw_rows = raw_df.count()

    mapping = build_column_mapping(raw_df.columns)
    renamed = raw_df
    for old_name, new_name in mapping.items():
        if old_name != new_name:
            renamed = renamed.withColumnRenamed(old_name, new_name)

    for required in EXPECTED_COLUMNS.keys():
        if required not in renamed.columns:
            renamed = renamed.withColumn(required, F.lit(None).cast("string"))

    trimmed = renamed.select(
        *[F.trim(F.col(c).cast("string")).alias(c) for c in EXPECTED_COLUMNS.keys()]
    )
    typed_df, invalid_rows, typed_rows = cast_and_validate(trimmed)
    deduped_df, duplicates_removed = deduplicate_by_day(typed_df)
    complete_df, missing_days_filled, min_dia, max_dia = add_missing_days(deduped_df)

    numeric_cols = ["inflacion", "tasa_interes", "precio_dolar"]
    nulls_before = count_nulls(complete_df, numeric_cols)

    imputed_counts: Dict[str, int] = {}
    working_df = complete_df
    for col_name in numeric_cols:
        working_df, count = impute_column(working_df, col_name)
        imputed_counts[col_name] = count

    outlier_report: Dict[str, Dict[str, float]] = {}
    for col_name in numeric_cols:
        working_df, count, low, high = winsorize_iqr(working_df, col_name)
        outlier_report[col_name] = {
            "detectados": count,
            "limite_inferior": low,
            "limite_superior": high,
        }

    final_df = (
        working_df.select(
            F.col("dia").cast("int"),
            F.col("inflacion").cast("double"),
            F.col("tasa_interes").cast("double"),
            F.col("precio_dolar").cast("double"),
            F.col("dia_insertado").cast("int"),
            F.col("imputado_inflacion").cast("int"),
            F.col("imputado_tasa_interes").cast("int"),
            F.col("imputado_precio_dolar").cast("int"),
            F.col("outlier_inflacion").cast("int"),
            F.col("outlier_tasa_interes").cast("int"),
            F.col("outlier_precio_dolar").cast("int"),
        )
        .orderBy("dia")
    )

    nulls_after = count_nulls(final_df, numeric_cols)
    final_rows = final_df.count()

    abs_output_dir = os.path.abspath(output_dir)
    parquet_path = os.path.join(abs_output_dir, "parquet")
    csv_path = os.path.join(abs_output_dir, "csv")

    final_df.write.mode("overwrite").parquet(parquet_path)
    final_df.coalesce(1).write.option("header", True).mode("overwrite").csv(csv_path)

    report = {
        "input_path": os.path.abspath(input_path),
        "filas_raw": raw_rows,
        "filas_tipadas": typed_rows,
        "filas_invalidas_eliminadas": invalid_rows,
        "duplicados_dia_eliminados": duplicates_removed,
        "rango_dia": {"min": min_dia, "max": max_dia},
        "dias_faltantes_rellenados": missing_days_filled,
        "nulos_antes_imputacion": nulls_before,
        "nulos_despues_imputacion": nulls_after,
        "imputaciones_por_columna": imputed_counts,
        "outliers_iqr": outlier_report,
        "filas_finales": final_rows,
        "output_parquet": parquet_path,
        "output_csv": csv_path,
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Limpieza completada.")
    print(f"Filas entrada: {raw_rows}")
    print(f"Filas finales: {final_rows}")
    print(f"CSV limpio: {csv_path}")
    print(f"Parquet limpio: {parquet_path}")
    print(f"Reporte: {os.path.abspath(report_path)}")

    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpieza profunda de dolar_data.csv con PySpark")
    parser.add_argument("--input", default="dolar_data.csv", help="Ruta del CSV de entrada.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "dolar_data_clean"),
        help="Directorio donde se escriben los datos limpios.",
    )
    parser.add_argument(
        "--report",
        default=os.path.join("output", "quality_report.json"),
        help="Ruta para guardar el reporte JSON de calidad.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output_dir, args.report)
