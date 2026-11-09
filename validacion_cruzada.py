r"""Comparación de clasificadores para predecir abandono (Dropout) en ActivaFit.

Ejemplo (PowerShell, desde la raíz del repositorio):
    .\.venv\Scripts\python.exe .\validacion_cruzada.py

Si el CSV está en otra ubicación:
    .\.venv\Scripts\python.exe .\validacion_cruzada.py --csv "C:\\ruta\\DadosV3.csv"

Dependencias: pandas, numpy y scikit-learn.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TARGET = "Dropout"
ID_COLUMN = "Id"
DATE_COLUMNS = (
    "EnrollmentStart",
    "EnrollmentFinish",
    "LastPeriodStart",
    "LastPeriodFinish",
    "DateLastVisit",
)

# Columnas que conviene revisar contra el momento previsto para emitir una alerta.
TEMPORAL_RISK_COLUMNS = (
    "EnrollmentFinish",
    "EnrollmentDuration",
    "EnrollmentDurationClass1",
    "EnrollmentDurationClass2",
    "LastPeriodStart",
    "LastPeriodFinish",
    "DateLastVisit",
    "DaysWithoutFrequency",
    "DaysWithoutFrequencyClass1",
    "DaysWithoutFrequencyClass2",
    "LifetimeValue",
    "LifetimeValueClass1",
    "AverageFrequency",
    "AverageFrequencyClass1",
    "AverageFrequencyClass2",
    "NumberOfFrequencies",
    "NumberOfFrequenciesClass1",
    "AttendedClasses",
    "AttendedClassesClass1",
    "AttendedClassesAverage",
    "AttendedClassesAverageClass1",
    "RealNumberOfVisits",
    "Ratio",
    "RatioClass2",
    "NumberOfRenewals",
    "EnrollmentStart",
    "AthleticsActivities",
    "WaterActivities",
    "FitnessActivities",
    "DanceActivities",
    "TeamActivities",
    "RacketActivities",
    "CombatActivities",
    "NatureActivities",
    "SpecialActivities",
    "OtherActivities",
    "NumberOfActivities",
    "HasReferences",
    "NumberOfReferences",
)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evalúa clasificadores con validación cruzada estratificada para Dropout."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path.home() / "Downloads" / "DadosV3.csv",
        help="CSV separado por punto y coma (por defecto: ~/Downloads/DadosV3.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_dir / "resultados" / "validacion_cruzada.csv",
        help="Ruta del CSV resumen (por defecto: resultados/validacion_cruzada.csv).",
    )
    return parser.parse_args()


def encode_target(values: pd.Series) -> pd.Series:
    """Normaliza Dropout a 0/1, aceptando enteros, booleanos o texto."""
    normalized: list[int | None] = []
    for value in values:
        label = str(value).strip().lower()
        if label in {"1", "true"}:
            normalized.append(1)
        elif label in {"0", "false"}:
            normalized.append(0)
        else:
            normalized.append(None)
    result = pd.Series(normalized, index=values.index, dtype="Int64")
    if result.isna().any():
        bad_values = values[result.isna()].unique().tolist()
        raise ValueError(f"Valores no reconocidos en {TARGET}: {bad_values}")
    return result.astype("int8")


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el CSV: {csv_path}\n"
            "Pasa su ubicación con --csv, por ejemplo: "
            'python validacion_cruzada.py --csv "C:\\ruta\\DadosV3.csv"'
        )

    # El archivo del proyecto usa punto y coma como separador.
    data = pd.read_csv(csv_path, sep=";")
    missing = [column for column in (ID_COLUMN, TARGET) if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing}. Columnas halladas: {data.columns.tolist()}")

    y = encode_target(data[TARGET])
    # Se conservan todos los predictores presentes; se excluyen solamente Id y Dropout.
    X = data.drop(columns=[ID_COLUMN, TARGET]).copy()

    # Convertir fechas a marcas numéricas es una transformación determinista; no estima
    # parámetros con datos de validación. La imputación/escalado/codificación sí se ajustan
    # dentro de cada fold mediante los Pipeline y ColumnTransformer de abajo.
    for column in DATE_COLUMNS:
        if column in X.columns:
            parsed = pd.to_datetime(X[column], errors="coerce")
            timestamps = parsed.astype("int64").astype("float64")
            timestamps[parsed.isna()] = np.nan
            X[column] = timestamps

    if y.nunique() != 2:
        raise ValueError(f"Se necesitan las clases 0 y 1 en {TARGET}; halladas: {sorted(y.unique().tolist())}")
    return X, y


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
    dense: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:  # Compatibilidad con versiones anteriores de scikit-learn.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=not dense)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0 if dense else 0.3,
    )


def build_models() -> dict[str, tuple[Any, bool, bool]]:
    """Devuelve estimador, si escala predictores numéricos e indicación de matriz densa."""
    return {
        "Dummy (most_frequent)": (DummyClassifier(strategy="most_frequent"), False, False),
        "Regresión logística": (LogisticRegression(max_iter=2000, random_state=42), True, False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True, False),
        "Gaussian Naive Bayes": (GaussianNB(), True, True),
        "Árbol de decisión": (DecisionTreeClassifier(random_state=42), False, False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), False, False),
        "SVM (RBF)": (SVC(kernel="rbf", random_state=42, cache_size=500), True, False),
    }


def main() -> None:
    args = parse_args()
    X, y = load_data(args.csv)
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    print(f"CSV: {args.csv}")
    print(f"Filas: {len(X):,} | Predictores: {X.shape[1]}")
    print(f"Dropout=0 (permanencia): {(y == 0).sum():,} | Dropout=1 (abandono): {(y == 1).sum():,}")
    print(f"Valores ausentes antes de imputar: {int(X.isna().sum().sum()):,}")
    print(f"Versiones: pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
    print("No se encontró partición train/test en los archivos del repositorio; se usa todo el CSV en CV.")
    risk_columns = [column for column in TEMPORAL_RISK_COLUMNS if column in X.columns]
    print("\nLimitación temporal: se conservaron predictores cuyo momento de disponibilidad debe verificarse:")
    print(", ".join(risk_columns) if risk_columns else "ninguno de la lista conocida")
    print(
        "Revise especialmente EnrollmentFinish, métricas de uso/inasistencia, actividades y referencias. "
        "No interprete el ranking como rendimiento de alerta temprana sin reconstruir los datos a una fecha de corte y evaluar por tiempo."
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "recall_abandono": make_scorer(recall_score, pos_label=1, zero_division=0),
        "precision_abandono": make_scorer(precision_score, pos_label=1, zero_division=0),
        "f1_abandono": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    rows: list[dict[str, float | str]] = []
    partial_output = args.output.with_name(f"{args.output.stem}_parcial{args.output.suffix}")
    try:
        for model_name, (estimator, scale_numeric, dense) in build_models().items():
            print(f"Iniciando: {model_name}", flush=True)
            pipeline = Pipeline(
                [
                    (
                        "preprocessing",
                        make_preprocessor(
                            numeric_features,
                            categorical_features,
                            scale_numeric=scale_numeric,
                            dense=dense,
                        ),
                    ),
                    ("classifier", estimator),
                ]
            )
            started = time.perf_counter()
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
                error_score="raise",
            )
            elapsed = time.perf_counter() - started

            row: dict[str, float | str] = {"modelo": model_name}
            for metric_name in scoring:
                fold_values = scores[f"test_{metric_name}"]
                row[f"{metric_name}_media"] = float(np.mean(fold_values))
                row[f"{metric_name}_desv"] = float(np.std(fold_values, ddof=1))
            row["fit_s_media"] = float(np.mean(scores["fit_time"]))
            row["fit_s_desv"] = float(np.std(scores["fit_time"], ddof=1))
            row["score_s_media"] = float(np.mean(scores["score_time"]))
            row["score_s_desv"] = float(np.std(scores["score_time"], ddof=1))
            row["tiempo_total_s"] = elapsed
            rows.append(row)
            # Mantiene un respaldo de los modelos finalizados mientras continúa la CV.
            checkpoint = pd.DataFrame(rows).sort_values("balanced_accuracy_media", ascending=False)
            partial_output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.to_csv(partial_output, index=False, encoding="utf-8-sig", float_format="%.6f")
            print(f"Completado: {model_name} ({elapsed:.1f} s)", flush=True)
    except KeyboardInterrupt:
        if rows:
            print(
                f"\nEjecución interrumpida. Los {len(rows)} modelos ya completados "
                f"quedaron guardados en: {partial_output.resolve()}"
            )
        else:
            print("\nEjecución interrumpida antes de completar un modelo; no hay resultados parciales.")
        raise SystemExit(130) from None

    summary = pd.DataFrame(rows).sort_values("balanced_accuracy_media", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.6f")
    partial_output.unlink(missing_ok=True)

    metric_columns = [
        "balanced_accuracy_media",
        "balanced_accuracy_desv",
        "recall_abandono_media",
        "recall_abandono_desv",
        "precision_abandono_media",
        "precision_abandono_desv",
        "f1_abandono_media",
        "f1_abandono_desv",
        "fit_s_media",
        "score_s_media",
        "tiempo_total_s",
    ]
    print("\nResumen (medias y desviaciones estándar entre folds; tiempos en segundos):")
    print(summary[["modelo", *metric_columns]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResultados guardados en: {args.output.resolve()}")
    print("La CV se ejecutó sobre el CSV completo. Si el proyecto incorpora un test separado, úselo solo al final.")


if __name__ == "__main__":
    main()
