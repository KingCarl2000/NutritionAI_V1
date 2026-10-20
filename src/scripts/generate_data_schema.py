import glob
import os
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
import yaml
from src.nutrition_core.logging.logger import logger

# Đường dẫn gốc của dự án (NutritionAI_V1)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def process_single_csv(file_path, table_name_override, config, existing_schema):
    """Xử lý một file CSV duy nhất và cập nhật vào existing_schema."""
    db_schema_name = config.get("db_schema", "raw")
    logger.info(
        f"Bắt đầu phân tích file: '{os.path.basename(file_path)}' thành bảng '{table_name_override}'"
    )

    try:
        df = pd.read_csv(file_path)
        logger.info(
            f"Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột."
        )
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {e} -> Bỏ qua.")
        return existing_schema

    schema_data = {
        "db_schema": db_schema_name,
        "table_name": table_name_override,
        "columns": {},
        "numerical_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "boolean_columns": [],
    }

    if not df.empty:
        for col_name, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "integer"
                schema_data["numerical_columns"].append(col_name)
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
                schema_data["numerical_columns"].append(col_name)
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "boolean"
                schema_data["boolean_columns"].append(col_name)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
                schema_data["datetime_columns"].append(col_name)
            else:
                col_type = "categorical"
                schema_data["categorical_columns"].append(col_name)

            schema_data["columns"][col_name] = {
                "type": col_type,
                "nullable": bool(df[col_name].isnull().any().item()),
                "description": "",
            }

    schema_data = {k: v for k, v in schema_data.items() if v != []}
    existing_schema[table_name_override] = schema_data
    return existing_schema


def generate_schemas(datasets_config: dict):
    """Tạo file YAML DB schema từ danh sách datasets_config truyền vào."""
    logger.info("=== Bắt đầu tạo YAML DB Schemas ===")
    yaml_groups = {}

    for ds_name, config in datasets_config.items():
        yaml_path = config["schema_path"]
        if yaml_path not in yaml_groups:
            yaml_groups[yaml_path] = {}

    for yaml_path in yaml_groups.keys():
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_groups[yaml_path] = yaml.safe_load(file) or {}
        else:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    for config_name, config in datasets_config.items():
        data_path = config["data_path"]
        schema_path = config["schema_path"]
        logger.info(f"=== Xử lý cấu hình DB: '{config_name}' ===")

        if not os.path.exists(data_path):
            logger.error(f"Đường dẫn không tồn tại: {data_path}")
            continue

        current_yaml_data = yaml_groups[schema_path]

        if os.path.isdir(data_path):
            logger.info("Phát hiện THƯ MỤC. Đang quét các file .csv...")
            folder_name = os.path.basename(os.path.normpath(data_path))
            if folder_name not in current_yaml_data:
                current_yaml_data[folder_name] = {}

            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            if not csv_files:
                logger.warning("Không có file .csv nào trong thư mục.")

            for file_path in csv_files:
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                current_yaml_data[folder_name] = process_single_csv(
                    file_path, table_name, config, current_yaml_data[folder_name]
                )

        elif os.path.isfile(data_path) and data_path.lower().endswith(".csv"):
            logger.info("Phát hiện FILE csv đơn lẻ.")
            if config_name not in current_yaml_data:
                current_yaml_data[config_name] = {}
            current_yaml_data[config_name] = process_single_csv(
                data_path, config_name, config, current_yaml_data[config_name]
            )
        else:
            logger.error("Đường dẫn không hợp lệ.")

        yaml_groups[schema_path] = current_yaml_data

    logger.info("--- HOÀN TẤT PHÂN TÍCH DB SCHEMA. ĐANG LƯU KẾT QUẢ ---")
    for yaml_path, yaml_data in yaml_groups.items():
        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(
                yaml_data,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info(f"Đã lưu YAML Schema tại: {yaml_path}")


def generate_pandera_ml_schema(
    ref_data_path: Path | str,
    output_schema_path: Path | str,
    pipeline_name: str = "ml_calories",
    target_col: str = "Calories",
):
    """Tạo file Python Pandera Schema linh hoạt dựa trên các tham số cấu hình truyền vào."""
    ref_data_path = Path(ref_data_path)
    output_schema_path = Path(output_schema_path)

    logger.info(
        f"=== Bắt đầu tạo Pandera Schema cho pipeline: '{pipeline_name}' ==="
    )

    if not ref_data_path.exists():
        logger.error(f"Không tìm thấy dữ liệu mẫu tại {ref_data_path}.")
        return

    try:
        df = pd.read_csv(ref_data_path)

        schema = pa.infer_schema(df)
        schema.coerce = True
        schema.strict = True

        raw_script = schema.to_script(minimal=True, script_type="model")

        # Tên class tương ứng với Pipeline (ví dụ: ml_calories -> MlCaloriesInputSchema / MlCaloriesTrainingSchema)
        model_name = "".join(
            [word.capitalize() for word in pipeline_name.split("_")]
        )
        input_schema_class = f"{model_name}InputSchema"
        training_schema_class = f"{model_name}TrainingSchema"

        lines = raw_script.split("\n")
        updated_lines = []
        target_line = ""

        for line in lines:
            if "class GeneratedModel(" in line:
                line = line.replace("GeneratedModel", input_schema_class)
            if target_col in line and "=" in line:
                target_line = line
                continue
            updated_lines.append(line)

        training_class_def = [
            "",
            "",
            f"class {training_schema_class}({input_schema_class}):",
            f'    """Schema dành riêng cho Training Pipeline (Kế thừa Input Schema + Cột Target)."""',
            f"    {target_line.strip()}" if target_line else "    pass",
        ]

        final_code = "\n".join(updated_lines + training_class_def)

        output_schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_schema_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        logger.info(
            f"✅ Đã tạo thành công Pandera Schema tại: {output_schema_path}"
        )

    except Exception as e:
        logger.error(f"Lỗi khi tạo Pandera schema cho {pipeline_name}: {e}")


if __name__ == "__main__":
    # =========================================================================
    # 1. CẤU HÌNH TẠO YAML DB SCHEMAS (Tùy chỉnh danh sách dataset tại đây)
    # =========================================================================
    DATASETS_CONFIG = {
        "calories": {
            "data_path": str(PROJECT_ROOT / "Data" / "raw" / "calories.csv"),
            "schema_path": str(
                PROJECT_ROOT / "src" / "api" / "data_schema" / "schema.yaml"
            ),
            "db_schema": "raw",
        },
        # Dễ dàng mở rộng thêm dataset khác tại đây trong tương lai:
        # "recipes": {
        #     "data_path": str(PROJECT_ROOT / "Data" / "raw" / "recipes.csv"),
        #     "schema_path": str(PROJECT_ROOT / "src" / "api" / "data_schema" / "recipes_schema.yaml"),
        #     "db_schema": "raw"
        # }
    }

    # =========================================================================
    # 2. CẤU HÌNH TẠO PANDERA ML SCHEMAS (Tùy chỉnh danh sách ML Pipelines tại đây)
    # =========================================================================
    ML_PIPELINES_CONFIG = [
        {
            "pipeline_name": "ml_calories",
            "target_col": "Calories",
            "ref_data_path": PROJECT_ROOT
            / "Artifacts"
            / "data"
            / "ml_calories"
            / "processed"
            / "train.csv",
            "output_schema_path": PROJECT_ROOT
            / "config"
            / "pipelines"
            / "ml_calories"
            / "calories_schema.py",
        },
        # Dễ dàng mở rộng thêm pipeline ML mới tại đây:
        # {
        #     "pipeline_name": "ml_recipe_recommendation",
        #     "target_col": "rating",
        #     "ref_data_path": PROJECT_ROOT / "Artifacts" / "data" / "ml_recipe" / "processed" / "train.csv",
        #     "output_schema_path": PROJECT_ROOT / "config" / "pipelines" / "ml_recipe" / "recipe_schema.py"
        # }
    ]

    # =========================================================================
    # 3. THỰC THI PIPELINE GENERATION
    # =========================================================================
    # Bỏ comment nếu muốn cập nhật cả Database YAML Schemas
    # generate_schemas(datasets_config=DATASETS_CONFIG)

    for ml_config in ML_PIPELINES_CONFIG:
        generate_pandera_ml_schema(
            ref_data_path=ml_config["ref_data_path"],
            output_schema_path=ml_config["output_schema_path"],
            pipeline_name=ml_config["pipeline_name"],
            target_col=ml_config["target_col"],
        )