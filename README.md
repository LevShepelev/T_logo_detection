README.md
T-Bank Logo Detector (zero-shot → small model → REST API)

Цель: детектировать логотип Т-Банка — стилизованная латинская буква “T” внутри щита (цвет любой) — и вернуть bbox’ы по API. Логотипы «Тинькофф» (wordmark) игнорируем.

Репозиторий включает:

Разметку (zero-shot) — пакет tbank-annotate на базе OWLv2/OWL-ViT с жёсткими промптами, негативными запросами и постпроцессингом (вычитание негативов, NMS, фильтры формы).

Тренировку «маленькой» модели (YOLO) на подготовленных лейблах.

REST API (FastAPI) по контракту, Docker-образ.

📦 Структура
tbank-logo/
├─ app/                           # REST API (FastAPI) — сервис /detect
│  ├─ main.py
│  └─ infer.py
│
├─ annotator/                     # Zero-shot разметка (пакет с CLI)
│  └─ tbank_annotate/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ detector.py
│     ├─ postprocess.py
│     ├─ prompts.py
│     └─ utils.py
│
├─ annotation/                    # Пайплайн подготовки датасета
│  ├─ run_annotate.sh             # пример запуска CLI
│  ├─ prepare_dataset_from_pseudo.py
│  └─ sanity_sample.py            # (опц.) выборка/коллажи для README
│
├─ training/                      # Тренировка маленькой модели
│  ├─ configs/
│  │  └─ data.yaml                # 1 класс: t_logo
│  ├─ train_small.py              # Ultralytics API
│  ├─ export_onnx.py              # (опц.) экспорт
│  └─ infer_benchmark.py          # (опц.) время/изобр.
│
├─ eval/
│  └─ eval_f1.py                  # F1@IoU=0.5 на валидном наборе
│
├─ data/                          # (не коммитим крупные файлы)
│  ├─ images/                     # сырые изображения
│  ├─ labels_yolo/                # псевдо-лейблы
│  └─ yolo/
│     ├─ images/{train,val}/
│     └─ labels/{train,val}/
│
├─ models/
│  └─ tbank_detector.pt           # итоговые веса детектора
├─ pyproject.toml                 # Poetry
├─ Dockerfile
└─ README.md

🔽 Данные

Скачайте архив (1.4 ГБ) и распакуйте в data/images/
Источник: https://data.tinkoff.ru/s/YsqPKQkapc5xKMb (password: 7*5\Lq=Oik).

🐍 Установка (Poetry)

Установите Poetry (рекомендуется через pipx):

pipx install poetry
# либо: curl -sSL https://install.python-poetry.org | python3 -


Создайте venv и установите базовые зависимости:

poetry env use python3.10   # или 3.11+
poetry install              # без групп train/ocr


⚠️ PyTorch/torchvision (GPU): у Poetry нет “официального” индекса для CUDA-колёс.
Установите их в venv отдельно под вашу CUDA:

# пример для CUDA 12.4
poetry run pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.5.1" "torchvision==0.20.1"


(Если нужна тренировка YOLO: добавьте группу train.)

При необходимости поставьте группу для тренировки:

poetry install --with train


Проверка:

poetry run python -c "import torch;print(torch.__version__, torch.version.cuda)"

🏷️ Zero-shot разметка (OWLv2/OWL-ViT)

Инструмент tbank-annotate уже входит как CLI-скрипт. Он:

использует строгие позитивные промпты (heater-shield + одна Latin T),

прогоняет негативные запросы (VK/круг/«яйцо МТС»/wordmark/монограммы),

вычитает негативы по IoU,

применяет NMS и shape-фильтры (минимум размера, диапазон aspect-ratio),

сохраняет YOLO-лейблы (класс 0) и только положительные превью.

Запуск (рекомендуемые параметры):

poetry run tbank-annotate \
  --input data/images \
  --out-labels data/labels_yolo \
  --out-viz data/viz_pos \
  --model google/owlv2-large-patch14-ensemble \
  --pos-set strict \
  --use-neg \
  --score 0.26 --neg-score 0.30 \
  --nms 0.5 --topk 150 \
  --min-side 16 --ar-min 0.6 --ar-max 1.4 \
  --neg-iou 0.4


Подбор порогов:

мало срабатываний → --score 0.22–0.24, --min-side 14, --ar-min 0.5 --ar-max 1.6;

много ложных кругов/«яиц»/монограмм → ужесточить --score до 0.28–0.32, --ar-min 0.7 --ar-max 1.3, поднять --neg-iou до 0.5.

Если OWLv2 всё ещё пропускает цель — можно добавить второй прогон с Grounding DINO 1.5 и объединить боксы (WBF), но это опционально.

🧰 Подготовка датасета

Разделяем на train/val и приводим к структуре YOLO:

poetry run python annotation/prepare_dataset_from_pseudo.py \
  --images data/images \
  --labels data/labels_yolo \
  --out data/yolo \
  --train-ratio 0.9


Файл training/configs/data.yaml описывает 1 класс t_logo.

🏋️ Тренировка «маленькой» модели

Используем Ultralytics (YOLOv8-s по умолчанию, img=896: хорошо ловит мелкие логотипы и укладывается в лимиты T4 16 GB).

poetry install --with train
poetry run python training/train_small.py \
  --data "$(pwd)/training/configs/data.yaml" \
  --model yolov8s.pt \
  --imgsz 896 --epochs 100 --batch 16 \
  --device 0


По завершении лучшие веса копируются в models/tbank_detector.pt.

📏 Валидация (F1@IoU=0.5)

Скрипт подбирает лучший conf (score-threshold) на data/yolo/val:

poetry run python eval/eval_f1.py \
  --weights models/tbank_detector.pt \
  --imgsz 896 \
  --conf-grid 0.10,0.60,0.02


Запомните conf с лучшим F1 — его же используем в REST-сервисе по умолчанию.

🌐 REST API (FastAPI)

Контракт:

class BoundingBox(BaseModel):
    x_min: int; y_min: int; x_max: int; y_max: int

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]


Запуск сервиса локально:

poetry run python -m app.main
# сервис на http://0.0.0.0:8000


Пример запроса:

curl -X POST "http://localhost:8000/detect" \
  -F "file=@/path/to/image.jpg"


Ответ:

{"detections":[{"bbox":{"x_min":412,"y_min":145,"x_max":538,"y_max":278}}]}


Ограничения: ≤10 с/изобр., GPU 16 GB (T4) — при imgsz≈896 и YOLOv8-s легко укладывается.

🐳 Docker

Сборка и запуск:

docker build -t tbank-detector .
docker run --gpus all -p 8000:8000 --rm tbank-detector