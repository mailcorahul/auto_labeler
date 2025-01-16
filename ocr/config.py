"""
models supported: [
    "microsoft/trocr-large-handwritten"
]
"""

OCR_CONFIG = {
    "auto_label_mode": "OCR",
    "model": "mindee/doctr",
    "text_detection": "db_resnet50",
    "text_recognition": "parseq"
}