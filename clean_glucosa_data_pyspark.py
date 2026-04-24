from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


EXPECTED_COLUMNS = ["edad", "imc", "actividad_fisica", "nivel_glucosa"]


ALIASES = {
    "edad": "edad",
    "age": "edad",
    "imc": "imc",
    "bmi": "imc",
    "actividad_fisica": "actividad_fisica",
    "actividadfisica": "actividad_fisica",
    "physical_activity": "actividad_fisica",
    "nivel_glucosa": "nivel_glucosa",
    "glucose_level": "nivel_glucosa",
}


def normalize_column_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_mapping(columns: List[str]) -> Dict[str, str]:
    mapped: Dict[str, str] = {}
    used = set()
    for col in columns:
        canonical = ALIASES.get(normalize_column_name(col), normalize_column_name(col))
        if canonical in used:
            continue
        mapped[col] = canonical
        used.add(canonical)
    return mapped


def percentile_pair(df: DataFrame, col_name: str) -> Tuple[float, float]:
    q = df.select(F.percentile_approx(F.col(col_name), [0.25, 0.75], 10000).alias("q")).first()["q"]
    return float(q[0]), float(q[1])


def percentile_single(df: DataFrame, col_name: str, p: float) -> float:
    return float(df.select(F.percentile_approx(F.col(col_name), p, 10000).alias("p")).first()["p"])


def count_nulls(df: DataFrame, cols: List[str]) -> Dict[str, int]:
    row = df.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]).first()
    return {c: int(row[c]) for c in cols}


def cast_and_validate(df: DataFrame) -> Tuple[DataFrame, int, int]:
    typed = df.select(
        F.regexp_replace(F.col("edad"), ",", ".").cast("int").alias("edad"),
        F.regexp_replace(F.col("imc"), ",", ".").cast("double").alias("imc"),
        F.regexp_replace(F.col("actividad_fisica"), ",", ".").cast("int").alias("actividad_fisica"),
        F.regexp_replace(F.col("nivel_glucosa"), ",", ".").cast("double").alias("nivel_glucosa"),
    )
    invalid = (
        F.col("edad").isNull()
        | F.col("imc").isNull()
        | F.col("actividad_fisica").isNull()
        | F.col("nivel_glucosa").isNull()
        | (F.col("edad") < 0)
        | (F.col("edad") > 120)
        | (F.col("imc") <= 0)
        | (F.col("imc") > 80)
        | (F.col("actividad_fisica") < 0)
        | (F.col("actividad_fisica") > 24)
        | (F.col("nivel_glucosa") < 20)
        | (F.col("nivel_glucosa") > 600)
    )
    invalid_count = typed.filter(invalid).count()
    return typed.filter(~invalid), invalid_count, typed.count()


def deduplicate_exact(df: DataFrame) -> Tuple[DataFrame, int]:
    before = df.count()
    dedup = df.dropDuplicates()
    removed = before - dedup.count()
    return dedup, removed


def impute_column(
    df: DataFrame, col_name: str, partition_cols: List[str], order_col: str
) -> Tuple[DataFrame, int]:
    median = percentile_single(df.filter(F.col(col_name).isNotNull()), col_name, 0.5)
    w_prev = Window.partitionBy(*partition_cols).orderBy(order_col).rowsBetween(Window.unboundedPreceding, 0)
    w_next = Window.partitionBy(*partition_cols).orderBy(order_col).rowsBetween(0, Window.unboundedFollowing)

    prev_col = f"_{col_name}_prev"
    next_col = f"_{col_name}_next"
    flag_col = f"imputado_{col_name}"

    staged = (
        df.withColumn(prev_col, F.last(F.col(col_name), ignorenulls=True).over(w_prev))
        .withColumn(next_col, F.first(F.col(col_name), ignorenulls=True).over(w_next))
        .withColumn(flag_col, F.col(col_name).isNull().cast("int"))
    )

    imputed = (
        staged.withColumn(
            col_name,
            F.when(
                F.col(col_name).isNull(),
                F.when(
                    F.col(prev_col).isNotNull() & F.col(next_col).isNotNull(),
                    (F.col(prev_col) + F.col(next_col)) / F.lit(2.0),
                ).otherwise(F.coalesce(F.col(prev_col), F.col(next_col), F.lit(median))),
            ).otherwise(F.col(col_name)),
        )
        .drop(prev_col, next_col)
    )
    count = imputed.filter(F.col(flag_col) == 1).count()
    return imputed, count


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


def run_pipeline(input_path: str, dataset_dir: str, cleaned_subdir: str, report_name: str) -> None:
    spark = SparkSession.builder.appName("LimpiezaProfundaGlucosa").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(input_path)
    )
    raw_rows = raw.count()

    mapping = build_mapping(raw.columns)
    renamed = raw
    for old, new in mapping.items():
        if old != new:
            renamed = renamed.withColumnRenamed(old, new)

    for required in EXPECTED_COLUMNS:
        if required not in renamed.columns:
            renamed = renamed.withColumn(required, F.lit(None).cast("string"))

    selected = renamed.select(*[F.trim(F.col(c).cast("string")).alias(c) for c in EXPECTED_COLUMNS])
    typed, invalid_removed, typed_rows = cast_and_validate(selected)
    deduped, duplicates_removed = deduplicate_exact(typed)

    numeric_cols = ["imc", "nivel_glucosa"]
    nulls_before = count_nulls(deduped, numeric_cols)

    working = deduped
    imputed_counts: Dict[str, int] = {}
    for col_name in numeric_cols:
        working, c = impute_column(
            working,
            col_name=col_name,
            partition_cols=["actividad_fisica"],
            order_col="edad",
        )
        imputed_counts[col_name] = c

    outliers_report: Dict[str, Dict[str, float]] = {}
    for col_name in numeric_cols:
        working, count, low, high = winsorize_iqr(working, col_name)
        outliers_report[col_name] = {
            "detectados": count,
            "limite_inferior": low,
            "limite_superior": high,
        }

    final_df = (
        working.select(
            F.col("edad").cast("int"),
            F.col("imc").cast("double"),
            F.col("actividad_fisica").cast("int"),
            F.col("nivel_glucosa").cast("double"),
            F.col("imputado_imc").cast("int"),
            F.col("imputado_nivel_glucosa").cast("int"),
            F.col("outlier_imc").cast("int"),
            F.col("outlier_nivel_glucosa").cast("int"),
        )
        .orderBy("edad", "actividad_fisica")
    )

    final_rows = final_df.count()
    nulls_after = count_nulls(final_df, numeric_cols)

    dataset_folder = os.path.abspath(dataset_dir)
    cleaned_folder = os.path.join(dataset_folder, cleaned_subdir)
    csv_path = os.path.join(cleaned_folder, "csv")
    parquet_path = os.path.join(cleaned_folder, "parquet")
    report_path = os.path.join(dataset_folder, report_name)

    final_df.write.mode("overwrite").parquet(parquet_path)
    final_df.coalesce(1).write.option("header", True).mode("overwrite").csv(csv_path)

    report = {
        "input_path": os.path.abspath(input_path),
        "filas_raw": raw_rows,
        "filas_tipadas": typed_rows,
        "filas_invalidas_eliminadas": invalid_removed,
        "duplicados_exactos_eliminados": duplicates_removed,
        "nulos_antes_imputacion": nulls_before,
        "nulos_despues_imputacion": nulls_after,
        "imputaciones_por_columna": imputed_counts,
        "outliers_iqr": outliers_report,
        "filas_finales": final_rows,
        "output_csv": csv_path,
        "output_parquet": parquet_path,
    }

    os.makedirs(dataset_folder, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Limpieza completada.")
    print(f"Filas entrada: {raw_rows}")
    print(f"Filas finales: {final_rows}")
    print(f"Carpeta dataset: {dataset_folder}")
    print(f"Carpeta limpio: {cleaned_folder}")
    print(f"Reporte: {report_path}")

    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpieza profunda de glucosa_data.csv con PySpark")
    parser.add_argument("--input", default="glucosa_data.csv", help="Ruta del CSV de entrada.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join("output", "glucosa_data"),
        help="Carpeta base dedicada al dataset.",
    )
    parser.add_argument(
        "--cleaned-subdir",
        default="limpio_glucosa_data",
        help="Subcarpeta dentro de dataset-dir para datos limpios.",
    )
    parser.add_argument(
        "--report-name",
        default="quality_report_glucosa.json",
        help="Nombre del reporte JSON dentro de dataset-dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.dataset_dir, args.cleaned_subdir, args.report_name)
