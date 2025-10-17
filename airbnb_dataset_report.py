#!/usr/bin/env python3
"""
airbnb_dataset_report.py

Usage:
  python airbnb_dataset_report.py --csv data/listings.csv --target room_type

Считает:
- N (строк), d (признаков), K (классов для target)
- % пропусков по всем ячейкам
- дисбаланс классов (распределение и max/min)
- % строк с выбросами (IQR) по числовым признакам
- количество признаков по типам: numeric / categorical / text / bool / datetime / other
- флаги да/нет: пропуски / разнородность / дисбаланс / много выбросов / текстовые признаки
Сохраняет: report.md и report_metrics.json
"""
import argparse, json, os, glob
import pandas as pd
import numpy as np

def guess_csv_path(csv_path_arg):
    if csv_path_arg: return csv_path_arg
    csvs = [p for p in glob.glob("*.csv")] + [p for p in glob.glob("**/*.csv", recursive=True)]
    if len(csvs) == 1: return csvs[0]
    if not csvs: raise FileNotFoundError("CSV не найден. Укажи --csv <path> или положи один *.csv в папку.")
    raise RuntimeError(f"Найдено несколько CSV: {csvs[:5]}... укажи --csv <path>.")

def split_object_columns(df):
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    text_cols, categorical_cols = [], []
    for col in obj_cols:
        s = df[col]
        try:
            s2 = s.dropna().astype(str)
            mean_len = (s2.str.len().mean() if len(s2) else 0) or 0
            space_ratio = (s2.str.contains(r"\s").mean() if len(s2) else 0) or 0
        except Exception:
            mean_len, space_ratio = 0, 0
        if mean_len >= 20 or space_ratio >= 0.5:
            text_cols.append(col)
        else:
            categorical_cols.append(col)
    return text_cols, categorical_cols

def compute_outlier_row_fraction(df_numeric: pd.DataFrame) -> float:
    if df_numeric.shape[1] == 0: return 0.0
    outlier_any = pd.Series(False, index=df_numeric.index)
    for col in df_numeric.columns:
        col_data = df_numeric[col].astype(float)
        col_nonnull = col_data.dropna()
        if col_nonnull.empty: continue
        q1, q3 = np.percentile(col_nonnull, 25), np.percentile(col_nonnull, 75)
        iqr = q3 - q1
        if iqr == 0:
            m, s = col_nonnull.mean(), col_nonnull.std(ddof=0)
            if s == 0: continue
            z = (col_data - m) / s
            mask = z.abs() > 3
        else:
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (col_data < low) | (col_data > high)
        outlier_any |= mask.fillna(False)
    return float(outlier_any.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Путь к CSV")
    ap.add_argument("--target", default="room_type", help="Целевой столбец (для классификации)")
    args = ap.parse_args()

    # читаем CSV (несколько попыток кодировки/сепаратора)
    csv_path = guess_csv_path(args.csv)
    last_err, df = None, None
    for sep in [",", ";", "\t"]:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] >= 5: break
            except Exception as e:
                last_err = e; df = None
        if df is not None: break
    if df is None:
        raise RuntimeError(f"Не удалось прочитать CSV. Последняя ошибка: {last_err}")

    N = int(len(df)); total_columns = int(df.shape[1])

    # типы признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()

    # догадаться о datetime в object-колонках (по сэмплу)
    add_dt = []
    for col in df.columns:
        if col in datetime_cols: continue
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str).head(200)
            if len(sample):
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                if parsed.notna().mean() >= 0.7: add_dt.append(col)
    datetime_cols = sorted(set(datetime_cols) | set(add_dt))
    # не даём им уйти в text/cat
    base = df.drop(columns=datetime_cols, errors='ignore')
    text_cols, categorical_cols = split_object_columns(base)

    accounted = set(numeric_cols) | set(bool_cols) | set(datetime_cols) | set(text_cols) | set(categorical_cols)
    other_cols = [c for c in df.columns if c not in accounted]

    total_cells = N * total_columns if N and total_columns else 0
    missing_pct = float((df.isna().sum().sum() / total_cells) * 100) if total_cells else 0.0
    has_missing = missing_pct > 0

    target = args.target if args.target in df.columns else None
    K, imbalance = None, None
    if target:
        vc = df[target].value_counts(dropna=True)
        K = int(vc.shape[0])
        perc = (vc / vc.sum() * 100.0).round(2)
        if len(perc) >= 2:
            imbalance = {
                "distribution_percent": perc.to_dict(),
                "max_class_percent": float(perc.max()),
                "min_class_percent": float(perc.min()),
                "max_to_min_ratio": round(float(perc.max() / max(perc.min(), 1e-9)), 2)
            }
        else:
            imbalance = {"distribution_percent": perc.to_dict()}

    outlier_rows_pct = compute_outlier_row_fraction(df.select_dtypes(include=[np.number])) * 100.0
    many_outliers = outlier_rows_pct >= 5.0
    heterogeneous = (len(numeric_cols) > 0) and (len(text_cols)+len(categorical_cols)+len(datetime_cols)+len(bool_cols)+len(other_cols) > 0)
    has_text = len(text_cols) > 0
    d = total_columns - (1 if target else 0)

    metrics = {
        "dataset_path": os.path.abspath(csv_path),
        "target_column": target,
        "N (rows)": N,
        "total_columns": total_columns,
        "d (features)": d,
        "K (classes)": K,
        "missing_pct_overall": round(missing_pct, 3),
        "class_imbalance": imbalance,
        "outlier_rows_pct_iqr": round(outlier_rows_pct, 2),
        "num_numeric_features": len(numeric_cols),
        "num_categorical_features": len(categorical_cols),
        "num_text_features": len(text_cols),
        "num_boolean_features": len(bool_cols),
        "num_datetime_features": len(datetime_cols),
        "num_other_features": len(other_cols),
        "other_feature_names": other_cols,
        "flags": {
            "has_missing": bool(has_missing),
            "heterogeneous_features": bool(heterogeneous),
            "imbalanced_classes": bool(imbalance and (imbalance.get("max_class_percent",0) >= 70 or imbalance.get("min_class_percent",0) <= 10)),
            "many_outliers": bool(many_outliers),
            "has_text_features": bool(has_text)
        }
    }

    with open("report_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    def yn(b): return "да" if b else "нет"
    lines = []
    lines += [f"# Отчёт по датасету",
              f"- Файл: `{os.path.basename(csv_path)}`",
              f"- Целевой столбец (target): `{target}`",
              f"\n## Ключевые параметры",
              f"- N (число объектов): **{N}**",
              f"- d (число признаков): **{d}** (всего столбцов: {total_columns})",
              f"- K (число классов): **{K}**" if K is not None else "- K (число классов): — (не задан target)",
              f"- Пропущенные значения (всего): **{round(missing_pct,3)}%**"]
    if imbalance:
        dist = ", ".join([f"{k}: {v}%" for k, v in imbalance.get("distribution_percent", {}).items()])
        lines += [f"\n## Несбалансированность классов",
                  f"- Распределение: {dist}",
                  f"- Отношение max/min: {imbalance.get('max_to_min_ratio', '—')}"]
    lines += [f"\n## Выбросы",
              f"- Доля строк с выбросами (IQR): **{round(outlier_rows_pct,2)}%**",
              f"\n## Типы признаков",
              f"- Числовые: {len(numeric_cols)}",
              f"- Категориальные: {len(categorical_cols)}",
              f"- Текстовые: {len(text_cols)}",
              f"- Булевы: {len(bool_cols)}",
              f"- Датавременные: {len(datetime_cols)}",
              f"- Другие ({len(other_cols)}): {other_cols}",
              f"\n## Флаги (да/нет)",
              f"- Наличие пропусков: **{yn(missing_pct>0)}**",
              f"- Разнородные признаки: **{yn(heterogeneous)}**",
              f"- Несбалансированные классы: **{yn(metrics['flags']['imbalanced_classes'])}** (критерий: >=70% или <=10%)",
              f"- Большое количество выбросов: **{yn(many_outliers)}** (порог: ≥5% строк)",
              f"- Наличие текстовых признаков: **{yn(has_text)}**"]
    with open("report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved: report_metrics.json, report.md")
    print("Done.")

if __name__ == "__main__":
    main()
