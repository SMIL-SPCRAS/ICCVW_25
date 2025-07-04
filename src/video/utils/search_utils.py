# coding: utf-8

import copy
import logging
import numpy as np
import datetime
from itertools import product
from typing import Any

def format_result_box_dual(step_num, param_name, candidate, fixed_params, dev_metrics, test_metrics, is_best=False):
    title = f"Шаг {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]

    def format_metrics_block(metrics, label):
        lines = [f"  Результаты ({label.upper()}):"]
        for k in ["uar", "war", "mf1", "wf1", "loss", "mean"]:
            if k in metrics:
                val = metrics[k]
                line = f"    {k.upper():12} = {val:.4f}" if isinstance(val, float) else f"    {k.upper():12} = {val}"
                if is_best and label.lower() == "dev" and k.lower() == "mean":
                    line += " ✅"
                lines.append(line)
        return lines

    content_lines = [title, "  Фиксировано:"]
    content_lines += [f"    {line}" for line in fixed_lines]

    # DEV блок
    content_lines += format_metrics_block(dev_metrics, "dev")
    content_lines.append("")

    # TEST блок
    content_lines += format_metrics_block(test_metrics, "test")

    # GAP
    if "mean" in dev_metrics and "mean" in test_metrics:
        gap_val = dev_metrics["mean"] - test_metrics["mean"]
        gap_str = f"    GAP          = {gap_val:+.4f}"
        content_lines.append(gap_str)

    max_width = max(len(line) for line in content_lines)
    border_top = "┌" + "─" * (max_width + 2) + "┐"
    border_bot = "└" + "─" * (max_width + 2) + "┘"

    box = [border_top]
    for line in content_lines:
        box.append(f"│ {line.ljust(max_width)} │")
    box.append(border_bot)

    return "\n".join(box)



def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
    csv_prefix: str = None
):
    current_best_params = copy.deepcopy(default_values)
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Жадный (поэтапный) перебор гиперпараметров (Dev-based) ===\n")
        f.write(f"Модель: {model_name}\n")

    for i, param_name in enumerate(all_param_names):
        candidates = param_grid[param_name]
        tried_value = current_best_params[param_name]

        if i == 0:
            candidates_to_try = candidates
        else:
            candidates_to_try = [v for v in candidates if v != tried_value]

        best_val_for_param = tried_value
        best_metric_for_param = float("-inf")

        # Если не первый шаг — вставим текущую комбу
        if i != 0:
            config_default = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config_default, k, v)
            logging.info(f"[ШАГ {i+1}] {param_name} = {tried_value} (ранее проверенный)")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{csv_prefix}_{model_name}_{param_name}_{tried_value}_{timestamp}.csv" if csv_prefix else None

            dev_mean_default, dev_metrics_default, test_metrics_default = train_fn(
                config_default,
                train_loader,
                dev_loader,
                test_loader,
                metrics_csv_path=csv_filename
            )

            box_text = format_result_box_dual(
                step_num=i+1,
                param_name=param_name,
                candidate=tried_value,
                fixed_params={k: v for k, v in current_best_params.items() if k != param_name},
                dev_metrics=dev_metrics_default,
                test_metrics=test_metrics_default,
                is_best=True
            )

            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")

            _log_dataset_metrics(dev_metrics_default, overrides_file, label="dev")
            _log_dataset_metrics(test_metrics_default, overrides_file, label="test")

            best_metric_for_param = dev_mean_default

        for candidate in candidates_to_try:
            config = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config, k, v)
            setattr(config, param_name, candidate)
            logging.info(f"[ШАГ {i+1}] {param_name} = {candidate}, (остальные {current_best_params})")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{csv_prefix}_{model_name}_{param_name}_{candidate}_{timestamp}.csv" if csv_prefix else None

            dev_mean, dev_metrics, test_metrics = train_fn(
                config,
                train_loader,
                dev_loader,
                test_loader,
                metrics_csv_path=csv_filename
            )

            is_better = dev_mean > best_metric_for_param
            box_text = format_result_box_dual(
                step_num=i+1,
                param_name=param_name,
                candidate=candidate,
                fixed_params={k: v for k, v in current_best_params.items() if k != param_name},
                dev_metrics=dev_metrics,
                test_metrics=test_metrics,
                is_best=is_better
            )

            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")

            _log_dataset_metrics(dev_metrics, overrides_file, label="dev")
            _log_dataset_metrics(test_metrics, overrides_file, label="test")

            if is_better:
                best_val_for_param = candidate
                best_metric_for_param = dev_mean

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n>> [Итог Шаг{i+1}]: Лучший {param_name}={best_val_for_param}, dev_mean={best_metric_for_param:.4f}\n")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Итоговая комбинация (Dev-based) ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")

    logging.info("Готово! Лучшие параметры подобраны.")


def exhaustive_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    csv_prefix: str = None
):
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Полный перебор гиперпараметров (Dev-based) ===\n")
        f.write(f"Модель: {model_name}\n")

    best_config = None
    best_metric = float("-inf")
    best_metrics = {}
    combo_id = 0

    for combo in product(*(param_grid[param] for param in all_param_names)):
        combo_id += 1
        param_combo = dict(zip(all_param_names, combo))

        config = copy.deepcopy(base_config)
        for k, v in param_combo.items():
            setattr(config, k, v)

        logging.info(f"\n[Комбинация #{combo_id}] {param_combo}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{csv_prefix}_{model_name}_combo{combo_id}_{timestamp}.csv" if csv_prefix else None

        dev_mean, dev_metrics, test_metrics = train_fn(
            config,
            train_loader,
            dev_loader,
            test_loader,
            metrics_csv_path=csv_filename
        )

        is_better = dev_mean > best_metric
        box_text = format_result_box_dual(
            step_num=combo_id,
            param_name=" + ".join(all_param_names),
            candidate=str(combo),
            fixed_params={},
            dev_metrics=dev_metrics,
            test_metrics=test_metrics,
            is_best=is_better
        )

        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box_text + "\n")

        _log_dataset_metrics(dev_metrics, overrides_file, label="dev")
        _log_dataset_metrics(test_metrics, overrides_file, label="test")

        if is_better:
            best_metric = dev_mean
            best_config = param_combo
            best_metrics = dev_metrics

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Лучшая комбинация (Dev-based) ===\n")
        for k, v in best_config.items():
            f.write(f"{k} = {v}\n")

    logging.info("Полный перебор завершён! Лучшие параметры выбраны.")
    return best_metric, best_config, best_metrics


def _compute_combined_avg(dev_metrics):
    if "by_dataset" not in dev_metrics:
        return None

    values = []
    for entry in dev_metrics["by_dataset"]:
        for key in ["uar", "war", "mf1", "wf1"]:
            if key in entry:
                values.append(entry[key])

    return float(np.mean(values)) if values else None


def _log_dataset_metrics(metrics, file_path, label="dev"):
    if "by_dataset" not in metrics:
        return

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Подробные метрики по каждому датасету ({label}):\n")
        for ds in metrics["by_dataset"]:
            name = ds.get("name", "unknown")
            f.write(f"  - {name}:\n")
            for k in ["loss", "uar", "war", "mf1", "wf1", "mean"]:
                if k in ds:
                    f.write(f"      {k.upper():4} = {ds[k]:.4f}\n")
        f.write(f"<<< Конец подробных метрик ({label})\n")
