r"""Selección F5 y evaluación final con separación entrenamiento/prueba.

Desde PowerShell, en la raíz del repositorio:
    .\.venv\Scripts\python.exe .\evaluacion_f5.py

Ruta alternativa del dataset:
    .\.venv\Scripts\python.exe .\evaluacion_f5.py --csv "C:\\ruta\\DadosV3.csv"

Las salidas nuevas se escriben en resultados_f5/. El CSV fuente, Id y Dropout
no se usan como variables predictoras. La evaluación se etiqueta como retrospectiva
porque el archivo no documenta un instante de predicción ni ventanas de observación.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
RETROSPECTIVE_LABEL = "Clasificación retrospectiva; disponibilidad temporal no comprobada"

# No se eliminan estas variables. Se nombran para revisión porque reflejan fechas,
# acumulados o actividad cuya disponibilidad antes del abandono no queda demostrada.
TEMPORAL_REVIEW_COLUMNS = (
    "EnrollmentStart",
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
    "UseByTime",
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
    "NumberOfFrequencies",
    "NumberOfFrequenciesClass1",
    "AttendedClasses",
    "AttendedClassesClass1",
    "AttendedClassesAverage",
    "AttendedClassesAverageClass1",
    "AllowedWeeklyVisitsBySLA",
    "AllowedNumberOfVisitsBySLA",
    "RealNumberOfVisits",
    "Ratio",
    "RatioClass2",
    "NumberOfRenewals",
    "HasReferences",
    "NumberOfReferences",
)

MODEL_EXPLANABILITY = {
    "Regresión logística": 3,
    "Árbol de decisión": 3,
    "KNN": 2,
    "Gaussian Naive Bayes": 2,
    "Random Forest": 1,
    "SVM (RBF)": 1,
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Selecciona el modelo para F5 con CV solo en train y prueba final reservada."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path.home() / "Downloads" / "DadosV3.csv",
        help="CSV separado por punto y coma (por defecto: ~/Downloads/DadosV3.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "resultados_f5",
        help="Carpeta para los artefactos nuevos (por defecto: resultados_f5/).",
    )
    return parser.parse_args()


def normalize_target(values: pd.Series) -> pd.Series:
    encoded: list[int | None] = []
    for value in values:
        label = str(value).strip().casefold()
        if label in {"1", "true"}:
            encoded.append(1)
        elif label in {"0", "false"}:
            encoded.append(0)
        else:
            encoded.append(None)
    target = pd.Series(encoded, index=values.index, dtype="Int64")
    if target.isna().any():
        bad = values[target.isna()].unique().tolist()
        raise ValueError(f"Valores no reconocidos en {TARGET}: {bad}")
    if target.nunique() != 2:
        raise ValueError(f"Se requieren ambas clases en {TARGET}; halladas: {target.unique().tolist()}")
    return target.astype("int8")


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Indica la ubicación con --csv."
        )
    data = pd.read_csv(csv_path, sep=";")
    required = [ID_COLUMN, TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing}; columnas halladas: {data.columns.tolist()}")

    y = normalize_target(data[TARGET])
    # Se conserva el conjunto de predictores del CSV, excluyendo solo el ID y el objetivo.
    X = data.drop(columns=[ID_COLUMN, TARGET]).copy()

    # Parseo determinista de fechas: no aprende parámetros de los folds ni del test.
    # Se mantienen las cinco columnas como variables numéricas; su validez temporal
    # requiere la fecha de corte de predicción, que el CSV no proporciona.
    for column in DATE_COLUMNS:
        if column in X.columns:
            parsed = pd.to_datetime(X[column], errors="coerce")
            timestamps = parsed.astype("int64").astype("float64")
            timestamps[parsed.isna()] = np.nan
            X[column] = timestamps
    return X, y


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
    dense: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:  # scikit-learn anterior a sparse_output
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=not dense)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0 if dense else 0.3,
    )


def make_pipeline(
    estimator: Any,
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
    dense: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessing",
                make_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    scale_numeric=scale_numeric,
                    dense=dense,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def model_specs() -> dict[str, tuple[Any, bool, bool]]:
    """Tuplas: estimador, escalar numéricas, producir matriz densa."""
    return {
        "Dummy (most_frequent)": (DummyClassifier(strategy="most_frequent"), False, False),
        "Regresión logística": (LogisticRegression(max_iter=2000, random_state=42), True, False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True, False),
        "Gaussian Naive Bayes": (GaussianNB(), True, True),
        "Árbol de decisión": (DecisionTreeClassifier(random_state=42), False, False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), False, False),
        "SVM (RBF)": (SVC(kernel="rbf", random_state=42, cache_size=500), True, False),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_abandono": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_abandono": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_abandono": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def cross_validate_with_separate_prediction_time(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    """CV manual para medir ajuste, predict y cálculo de métricas por separado."""
    folds: list[dict[str, float | str]] = []
    for fold_number, (fit_indices, validation_indices) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train.iloc[validation_indices]

        fold_model = clone(pipeline)
        fit_started = time.perf_counter()
        fold_model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - fit_started

        prediction_started = time.perf_counter()
        predictions = fold_model.predict(X_validation)
        prediction_seconds = time.perf_counter() - prediction_started

        evaluation_started = time.perf_counter()
        metrics = classification_metrics(y_validation, predictions)
        evaluation_seconds = time.perf_counter() - evaluation_started

        folds.append(
            {
                "modelo": model_name,
                "pliegue": fold_number,
                **metrics,
                "fit_time_s": fit_seconds,
                # Incluye el transform del Pipeline y predict del estimador, nada de fit ni métricas.
                "prediction_time_s": prediction_seconds,
                "metric_evaluation_time_s": evaluation_seconds,
            }
        )

    fold_frame = pd.DataFrame(folds)
    result: dict[str, float | str] = {"modelo": model_name, "n_folds": len(folds)}
    result["interpretacion"] = RETROSPECTIVE_LABEL
    for metric in ("balanced_accuracy", "recall_abandono", "precision_abandono", "f1_abandono"):
        result[f"{metric}_media"] = float(fold_frame[metric].mean())
        result[f"{metric}_desv"] = float(fold_frame[metric].std(ddof=1))
    for timing in ("fit_time_s", "prediction_time_s", "metric_evaluation_time_s"):
        result[f"{timing}_media"] = float(fold_frame[timing].mean())
        result[f"{timing}_desv"] = float(fold_frame[timing].std(ddof=1))
    return result, folds


def score_accuracy(value: float) -> int:
    if value >= 0.90:
        return 3
    if value >= 0.75:
        return 2
    return 1


def score_stability(std_value: float) -> int:
    if std_value <= 0.01:
        return 3
    if std_value <= 0.03:
        return 2
    return 1


def score_prediction_time(seconds_per_fold: float) -> int:
    if seconds_per_fold <= 0.05:
        return 3
    if seconds_per_fold <= 0.20:
        return 2
    return 1


def build_selection_matrix(cv_summary: pd.DataFrame) -> pd.DataFrame:
    matrix = cv_summary[cv_summary["modelo"].isin(MODEL_EXPLANABILITY)].copy()
    matrix["puntaje_metrica"] = matrix["balanced_accuracy_media"].map(score_accuracy)
    matrix["puntaje_estabilidad"] = matrix["balanced_accuracy_desv"].map(score_stability)
    matrix["puntaje_explicabilidad"] = matrix["modelo"].map(MODEL_EXPLANABILITY)
    matrix["puntaje_tiempo_prediccion"] = matrix["prediction_time_s_media"].map(score_prediction_time)
    matrix["total_ponderado"] = (
        matrix["puntaje_metrica"] * 0.40
        + matrix["puntaje_estabilidad"] * 0.20
        + matrix["puntaje_explicabilidad"] * 0.25
        + matrix["puntaje_tiempo_prediccion"] * 0.15
    )
    matrix["explicabilidad_criterio"] = matrix["modelo"].map(
        {
            "Regresión logística": "3: coeficientes globales interpretables",
            "Árbol de decisión": "3: reglas de decisión inspeccionables",
            "KNN": "2: explica por vecinos, aunque depende de casos y escala",
            "Gaussian Naive Bayes": "2: probabilidades por predictor; supuesto de independencia",
            "Random Forest": "1: conjunto de árboles menos transparente globalmente",
            "SVM (RBF)": "1: frontera no lineal de difícil interpretación directa",
        }
    )
    matrix = matrix.sort_values(
        ["total_ponderado", "balanced_accuracy_media", "puntaje_explicabilidad"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    matrix.insert(0, "puesto", np.arange(1, len(matrix) + 1))
    matrix["interpretacion"] = RETROSPECTIVE_LABEL
    return matrix


def write_temporal_limitations(
    output_path: Path,
    predictors: list[str],
) -> None:
    reviewed = [name for name in TEMPORAL_REVIEW_COLUMNS if name in predictors]
    lines = [
        "LIMITACIÓN DE DISPONIBILIDAD TEMPORAL — ACTivaFit / Dropout",
        "",
        "Etiqueta aplicada a todos los resultados: clasificación retrospectiva; disponibilidad temporal no comprobada.",
        "El CSV no define el instante en que se emitiría la predicción, una fecha de corte ni la ventana de cálculo de sus variables.",
        "Por eso no es posible demostrar qué campos estarían disponibles antes del abandono. No se excluyeron variables salvo Id y Dropout.",
        "",
        "Variables que requieren validación explícita con el proceso de negocio:",
        *[f"- {column}" for column in reviewed],
        "",
        "Revisión prioritaria: EnrollmentFinish, LastPeriodFinish, DateLastVisit, DaysWithoutFrequency, EnrollmentDuration,",
        "LifetimeValue, AverageFrequency, variables de asistencia y visitas, NumberOfRenewals, actividades acumuladas y referencias.",
        "Pueden ser valores calculados al cierre del período, depender de inactividad posterior o resumir un lapso que incluye el abandono.",
        "EnrollmentStart y fechas de visitas/periodos pueden ser anteriores al resultado, pero sin fecha de corte tampoco se confirma su disponibilidad.",
        "",
        "No presentar estas métricas como alerta temprana. Para ello, definir una fecha de corte operacional, reconstruir cada predictor usando",
        "solo información disponible hasta esa fecha y validar respetando el orden temporal (y, si corresponde, separando socios).",
        "El parseo de las cinco fechas a marcas numéricas es determinista y no aprende del test; esas columnas siguen incluidas como predictores.",
        "",
        "La partición aleatoria estratificada puede compartir periodos o socios entre train y test si el CSV tiene filas relacionadas; el CSV no permite verificarlo.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    X, y = load_data(args.csv)
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    # Se crea una sola partición estratificada; el test queda apartado hasta el ganador.
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    X_train, X_test = X.iloc[train_indices].copy(), X.iloc[test_indices].copy()
    y_train, y_test = y.iloc[train_indices].copy(), y.iloc[test_indices].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary_rows: list[dict[str, float | str]] = []
    fold_rows: list[dict[str, float | str]] = []
    specs = model_specs()

    print(f"CSV fuente: {args.csv}")
    print(f"Filas: {len(X):,} | Predictores: {X.shape[1]} (se excluyen solo {ID_COLUMN} y {TARGET})")
    print(f"Train: {len(X_train):,} (80 %) | Test reservado: {len(X_test):,} (20 %)")
    print(
        f"Train Dropout=0: {(y_train == 0).sum():,} | Dropout=1: {(y_train == 1).sum():,}"
    )
    print(
        f"Test Dropout=0: {(y_test == 0).sum():,} | Dropout=1: {(y_test == 1).sum():,}"
    )
    print("CV: StratifiedKFold(5, shuffle=True, random_state=42), únicamente sobre train.")
    print(f"Interpretación: {RETROSPECTIVE_LABEL}.")
    print("Revisión temporal completa: resultados_f5/limitaciones_temporales.txt")

    for model_name, (estimator, scale_numeric, dense) in specs.items():
        print(f"CV train: {model_name}...", flush=True)
        pipeline = make_pipeline(
            estimator,
            numeric_columns,
            categorical_columns,
            scale_numeric=scale_numeric,
            dense=dense,
        )
        aggregate, folds = cross_validate_with_separate_prediction_time(
            model_name,
            pipeline,
            X_train,
            y_train,
            cv,
        )
        cv_summary_rows.append(aggregate)
        fold_rows.extend(folds)

    cv_summary = pd.DataFrame(cv_summary_rows)
    selection = build_selection_matrix(cv_summary)
    winner_name = str(selection.iloc[0]["modelo"])
    cv_summary = cv_summary.sort_values("balanced_accuracy_media", ascending=False).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = args.output_dir / "metricas_cv_entrenamiento.csv"
    fold_path = args.output_dir / "metricas_por_pliegue.csv"
    selection_path = args.output_dir / "matriz_seleccion_f5.csv"
    final_metrics_path = args.output_dir / "evaluacion_final_test_ganador.csv"
    confusion_path = args.output_dir / "matriz_confusion_test_ganador.csv"
    limitations_path = args.output_dir / "limitaciones_temporales.txt"

    # Guardar CV y selección antes de mirar el conjunto de prueba.
    cv_summary.to_csv(cv_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    write_temporal_limitations(limitations_path, X.columns.tolist())

    print("\nCV de entrenamiento: medias y desviaciones estándar; tiempos promediados por pliegue (s).")
    cv_display = cv_summary[
        [
            "modelo",
            "balanced_accuracy_media",
            "balanced_accuracy_desv",
            "recall_abandono_media",
            "recall_abandono_desv",
            "precision_abandono_media",
            "precision_abandono_desv",
            "f1_abandono_media",
            "f1_abandono_desv",
            "fit_time_s_media",
            "prediction_time_s_media",
            "metric_evaluation_time_s_media",
        ]
    ]
    print(cv_display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nMatriz F5 (Dummy se excluye de la selección):")
    selection_display = selection[
        [
            "puesto",
            "modelo",
            "puntaje_metrica",
            "puntaje_estabilidad",
            "puntaje_explicabilidad",
            "puntaje_tiempo_prediccion",
            "total_ponderado",
            "balanced_accuracy_media",
            "balanced_accuracy_desv",
            "prediction_time_s_media",
        ]
    ]
    print(selection_display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nGanador por matriz F5 (solo CV de train): {winner_name}")

    # Desde aquí se evalúa exactamente un modelo y una vez en el test reservado.
    winner_estimator, winner_scale, winner_dense = specs[winner_name]
    final_pipeline = make_pipeline(
        clone(winner_estimator),
        numeric_columns,
        categorical_columns,
        scale_numeric=winner_scale,
        dense=winner_dense,
    )
    final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test)
    test_scores = classification_metrics(y_test, test_predictions)
    test_row = {
        "modelo_ganador_seleccionado_solo_con_cv": winner_name,
        **test_scores,
        "n_test": len(y_test),
        "interpretacion": RETROSPECTIVE_LABEL,
    }
    pd.DataFrame([test_row]).to_csv(final_metrics_path, index=False, encoding="utf-8-sig", float_format="%.8f")

    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    confusion_frame = pd.DataFrame(
        cm,
        index=["real_0_permanencia", "real_1_abandono"],
        columns=["predicho_0_permanencia", "predicho_1_abandono"],
    )
    confusion_frame.index.name = "clase_real"
    confusion_frame.to_csv(confusion_path, encoding="utf-8-sig")

    print("\nEvaluación final única del ganador en el test reservado (no se usó para selección):")
    for metric_name, value in test_scores.items():
        print(f"{metric_name}: {value:.4f}")
    print("Matriz de confusión (filas=real 0/1; columnas=predicho 0/1):")
    print(confusion_frame.to_string())
    print("\nArchivos generados:")
    for output in (cv_path, fold_path, selection_path, final_metrics_path, confusion_path, limitations_path):
        print(f"- {output.resolve()}")


if __name__ == "__main__":
    main()
