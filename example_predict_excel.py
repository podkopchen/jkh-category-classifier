"""
example_predict_excel.py

Скрипт для локальной проверки модели классификации обращений ЖКХ.

Что делает:
1. Читает Excel-файл с реальными обращениями.
2. Берет текст из столбца "Описание проблемы".
3. Предсказывает категорию с помощью обученной модели.
4. Сравнивает предсказание с исходной категорией из столбца "Категория".
5. Сохраняет несколько Excel-отчетов для анализа качества модели.

Ожидаемая структура проекта:

Category_DeepPavlov/
├── data/
│   └── jkh_requests.xlsx
├── outputs/
├── predict_model/
├── predict_excel.py
├── README.md
└── requirements.txt

Перед запуском установите зависимости:

pip install torch transformers pandas openpyxl scikit-learn tqdm joblib safetensors

Запуск:

python example_predict_excel.py
"""

from pathlib import Path

import joblib
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Пути проекта

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH = BASE_DIR / "data" / "jkh_requests.xlsx"
MODEL_DIR = BASE_DIR / "predict_model"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_EXCEL_PATH = OUTPUT_DIR / "predictions.xlsx"
OUTPUT_ERRORS_PATH = OUTPUT_DIR / "errors.xlsx"
OUTPUT_REPORT_PATH = OUTPUT_DIR / "classification_report.xlsx"
OUTPUT_CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.xlsx"
OUTPUT_ERROR_PAIRS_PATH = OUTPUT_DIR / "error_pairs.xlsx"
OUTPUT_HIGH_CONF_ERRORS_PATH = OUTPUT_DIR / "high_confidence_errors.xlsx"


# Названия колонок в исходной Excel-таблице

TEXT_COLUMN = "Описание проблемы"
TRUE_LABEL_COLUMN = "Категория"


# Параметры инференса

MAX_LENGTH = 128
BATCH_SIZE = 128

# Ошибки с такой уверенностью будут дополнительно сохранены в отдельный файл.
HIGH_CONFIDENCE_THRESHOLD = 90.0

# Если в исходной таблице есть категории, которых нет в модели,
# их можно исключить из отдельной "честной" метрики.
# Например, модель не обучалась на категории "Другое", поэтому полезно считать качество без нее.
EXCLUDED_FROM_KNOWN_ACCURACY = {"Другое"}


def check_required_files() -> None:
    """Проверяет, что все необходимые файлы лежат на своих местах."""

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найден Excel-файл: {EXCEL_PATH}\n"
            f"Положите исходную таблицу в папку data/."
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка модели: {MODEL_DIR}\n"
            f"Положите файлы модели в папку predict_model/."
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Не найден label_encoder.pkl: {LABEL_ENCODER_PATH}\n"
            f"Файл нужен для преобразования LABEL_0, LABEL_1 и т.д. в нормальные названия категорий."
        )


def load_model():
    """Загружает tokenizer, модель и label_encoder."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Устройство для расчета: {device}")

    print("Загружаю tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Загружаю модель...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print("Загружаю label_encoder...")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print("\nКлассы модели:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index:02d}. {class_name}")

    return tokenizer, model, label_encoder, device


def load_excel() -> pd.DataFrame:
    """
    Загружает Excel-файл и подготавливает рабочую таблицу.

    На выходе остается только две колонки:
    - Проблема
    - Изначальная Категория
    """

    print(f"\nЧитаю Excel-файл: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)

    print("\nКолонки в исходном файле:")
    for column in df.columns:
        print(f"- {column}")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"В таблице нет столбца '{TEXT_COLUMN}'.\n"
            f"Проверьте название столбца с текстом обращения."
        )

    if TRUE_LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"В таблице нет столбца '{TRUE_LABEL_COLUMN}'.\n"
            f"Проверьте название столбца с исходной категорией."
        )

    work_df = df[[TEXT_COLUMN, TRUE_LABEL_COLUMN]].copy()

    work_df = work_df.rename(
        columns={
            TEXT_COLUMN: "Проблема",
            TRUE_LABEL_COLUMN: "Изначальная Категория",
        }
    )

    # Чистим текст и категории от пустых значений и лишних пробелов.
    # Это важно, потому что строка "Системы ДУ и ППА " с пробелом в конце
    # будет считаться другой категорией, чем "Системы ДУ и ППА".
    work_df["Проблема"] = (
        work_df["Проблема"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    work_df["Изначальная Категория"] = (
        work_df["Изначальная Категория"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Пустые описания не проверяем, потому что модели нечего классифицировать.
    work_df = work_df[work_df["Проблема"] != ""].copy()
    work_df = work_df.reset_index(drop=True)

    print(f"\nСтрок для проверки: {len(work_df)}")

    return work_df


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    label_encoder,
    device,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[str], list[float]]:
    """
    Предсказывает категории пачками.

    Батчи нужны, чтобы не гонять модель по одной строке.
    На CPU это все равно может быть не очень быстро, но намного лучше, чем построчный режим.

    Возвращает:
    - список предсказанных категорий;
    - список уверенностей модели в процентах.
    """

    predicted_labels = []
    confidences_percent = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Предсказание"):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        batch_confidences, batch_pred_ids = torch.max(probabilities, dim=1)

        batch_pred_ids = batch_pred_ids.cpu().numpy()
        batch_confidences = batch_confidences.cpu().numpy()

        batch_labels = label_encoder.inverse_transform(batch_pred_ids)

        predicted_labels.extend(batch_labels)
        confidences_percent.extend(
            round(float(confidence) * 100, 2)
            for confidence in batch_confidences
        )

    return predicted_labels, confidences_percent


def build_result_table(work_df: pd.DataFrame, predicted_labels: list[str], confidences: list[float]) -> pd.DataFrame:
    """Собирает итоговую таблицу с исходной категорией, категорией ИИ и признаком совпадения."""

    result_df = work_df.copy()

    result_df["ИИ категория"] = predicted_labels
    result_df["ИИ категория"] = result_df["ИИ категория"].astype(str).str.strip()

    result_df["Уверенность (%)"] = confidences

    result_df["Совпало"] = (
        result_df["Изначальная Категория"] == result_df["ИИ категория"]
    )

    result_df = result_df[
        [
            "Проблема",
            "Изначальная Категория",
            "ИИ категория",
            "Уверенность (%)",
            "Совпало",
        ]
    ].copy()

    return result_df


def save_main_outputs(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сохраняет основной файл с предсказаниями и отдельный файл с ошибками."""

    result_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\nОсновной файл сохранен: {OUTPUT_EXCEL_PATH}")

    errors_df = result_df[result_df["Совпало"] == False].copy()
    errors_df = errors_df.sort_values("Уверенность (%)", ascending=False)

    errors_df.to_excel(OUTPUT_ERRORS_PATH, index=False)
    print(f"Файл с ошибками сохранен: {OUTPUT_ERRORS_PATH}")

    high_conf_errors_df = errors_df[
        errors_df["Уверенность (%)"] >= HIGH_CONFIDENCE_THRESHOLD
    ].copy()

    high_conf_errors_df.to_excel(OUTPUT_HIGH_CONF_ERRORS_PATH, index=False)
    print(
        f"Ошибки с уверенностью >= {HIGH_CONFIDENCE_THRESHOLD:.0f}% "
        f"сохранены: {OUTPUT_HIGH_CONF_ERRORS_PATH}"
    )

    return errors_df, high_conf_errors_df


def print_main_metrics(result_df: pd.DataFrame) -> None:
    """Печатает основные метрики в консоль."""

    total_count = len(result_df)
    correct_count = int(result_df["Совпало"].sum())
    error_count = total_count - correct_count

    accuracy = correct_count / total_count if total_count else 0

    print("\nОбщий результат")
    print(f"Всего строк: {total_count}")
    print(f"Правильно: {correct_count}")
    print(f"Ошибок: {error_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    known_df = result_df[
        ~result_df["Изначальная Категория"].isin(EXCLUDED_FROM_KNOWN_ACCURACY)
    ].copy()

    if len(known_df) != len(result_df):
        known_accuracy = known_df["Совпало"].mean() if len(known_df) else 0

        excluded_count = len(result_df) - len(known_df)

        print("\nAccuracy без исключенных категорий")
        print(f"Исключенные категории: {', '.join(sorted(EXCLUDED_FROM_KNOWN_ACCURACY))}")
        print(f"Исключено строк: {excluded_count}")
        print(f"Осталось строк: {len(known_df)}")
        print(f"Accuracy без исключенных категорий: {known_accuracy:.4f}")
        print(f"Accuracy без исключенных категорий (%): {known_accuracy * 100:.2f}%")


def save_classification_report(result_df: pd.DataFrame) -> pd.DataFrame:
    """Сохраняет classification report по всем категориям."""

    true_labels = result_df["Изначальная Категория"].astype(str)
    predicted_labels = result_df["ИИ категория"].astype(str)

    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T
    report_df.to_excel(OUTPUT_REPORT_PATH)

    print(f"\nОтчет по категориям сохранен: {OUTPUT_REPORT_PATH}")

    weak_categories = (
        report_df
        .drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        .sort_values("f1-score")
        .head(15)
    )

    print("\nСамые слабые категории по F1-score:")
    print(weak_categories)

    return report_df


def save_confusion_matrix(result_df: pd.DataFrame) -> pd.DataFrame:
    """Сохраняет матрицу ошибок."""

    true_labels = result_df["Изначальная Категория"].astype(str)
    predicted_labels = result_df["ИИ категория"].astype(str)

    all_labels = sorted(
        set(true_labels.tolist()) | set(predicted_labels.tolist())
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=all_labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true: {label}" for label in all_labels],
        columns=[f"pred: {label}" for label in all_labels],
    )

    matrix_df.to_excel(OUTPUT_CONFUSION_PATH)

    print(f"Матрица ошибок сохранена: {OUTPUT_CONFUSION_PATH}")

    return matrix_df


def save_error_pairs(errors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сохраняет самые частые пары ошибок.

    Это один из самых полезных отчетов:
    он показывает, какие категории модель путает чаще всего.
    """

    if errors_df.empty:
        print("\nОшибок нет, файл с парами ошибок не создан.")
        return pd.DataFrame()

    error_pairs_df = (
        errors_df
        .groupby(["Изначальная Категория", "ИИ категория"])
        .size()
        .reset_index(name="Количество")
        .sort_values("Количество", ascending=False)
    )

    error_pairs_df.to_excel(OUTPUT_ERROR_PAIRS_PATH, index=False)

    print(f"\nПары ошибок сохранены: {OUTPUT_ERROR_PAIRS_PATH}")

    print("\nСамые частые пары ошибок:")
    print(error_pairs_df.head(30))

    return error_pairs_df


def print_files_summary() -> None:
    """Печатает список созданных файлов."""

    print("\nГотово. Созданы файлы:")

    output_files = [
        OUTPUT_EXCEL_PATH,
        OUTPUT_ERRORS_PATH,
        OUTPUT_HIGH_CONF_ERRORS_PATH,
        OUTPUT_REPORT_PATH,
        OUTPUT_CONFUSION_PATH,
        OUTPUT_ERROR_PAIRS_PATH,
    ]

    for file_path in output_files:
        if file_path.exists():
            print(f"- {file_path}")


def main() -> None:
    """Основной сценарий работы скрипта."""

    check_required_files()

    tokenizer, model, label_encoder, device = load_model()

    work_df = load_excel()

    texts = work_df["Проблема"].tolist()

    predicted_labels, confidences = predict_batch(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        label_encoder=label_encoder,
        device=device,
        batch_size=BATCH_SIZE,
    )

    result_df = build_result_table(
        work_df=work_df,
        predicted_labels=predicted_labels,
        confidences=confidences,
    )

    errors_df, _ = save_main_outputs(result_df)

    print_main_metrics(result_df)

    save_classification_report(result_df)
    save_confusion_matrix(result_df)
    save_error_pairs(errors_df)

    print_files_summary()


if __name__ == "__main__":
    main()
