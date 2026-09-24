"""LLM Guard NER config for Horizon-Labs PII Redactor (use with llm_guard.input_scanners.Anonymize)."""
from llm_guard.model import Model

MODEL_TO_PRESIDIO = {
    "PERSON": "PERSON", "EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER", "LOCATION": "LOCATION",
    "STREET_ADDRESS": "LOCATION", "POSTCODE": "LOCATION", "ORGANIZATION": "ORGANIZATION", "DATE": "DATE_TIME",
    "DATE_OF_BIRTH": "DATE_TIME", "CREDIT_CARD": "CREDIT_CARD", "CREDIT_CARD_CVV": "CREDIT_CARD",
    "BANK_ACCOUNT": "IBAN_CODE", "IP_ADDRESS": "IP_ADDRESS", "URL": "URL", "NATIONAL_ID": "US_SSN",
    "TAX_ID": "US_SSN", "PASSPORT": "US_PASSPORT", "DRIVER_LICENSE": "US_DRIVER_LICENSE", "MEDICAL_ID": "MEDICAL_LICENSE",
    "USERNAME": "PERSON", "PASSWORD": "CRYPTO", "SECRET": "CRYPTO", "ACCOUNT_ID": "US_BANK_NUMBER",
    "AGE": "O", "COORDINATE": "LOCATION", "MAC_ADDRESS": "IP_ADDRESS", "DEVICE_ID": "O", "VEHICLE_ID": "O",
    "LICENSE_NUMBER": "US_DRIVER_LICENSE",
}


def horizon_pii_conf(size="small"):
    repo = f"Horizon-Labs/pii-redactor-{size}"
    return {
        "PRESIDIO_SUPPORTED_ENTITIES": sorted(set(v for v in MODEL_TO_PRESIDIO.values() if v != "O")),
        "DEFAULT_MODEL": Model(path=repo, onnx_path=repo, onnx_subfolder="onnx",
                               pipeline_kwargs={"aggregation_strategy": "simple"},
                               tokenizer_kwargs={"model_input_names": ["input_ids", "attention_mask"]}),
        "LABELS_TO_IGNORE": ["O"],
        "DEFAULT_EXPLANATION": f"Identified as {{}} by the {repo} model",
        "MODEL_TO_PRESIDIO_MAPPING": MODEL_TO_PRESIDIO,
        "CHUNK_OVERLAP_SIZE": 40,
        "CHUNK_SIZE": 600,
        "ID_SCORE_MULTIPLIER": 0.4,
        "ID_ENTITY_NAME": "ID",
    }
