"""Streamlit application for classifying and locating one egg per image."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path(
    os.environ.get("OBJDET_MODEL_PATH", APP_DIR / "sth_as-hw-tt.keras")
).expanduser()

# The original model uses output 0 for the background ("sth") and output 1
# for "as". Outputs 2 and 3 are the two additional foreground classes.
CLASS_CONFIG = (
    {"index": 1, "label": "as", "color": (46, 204, 113)},
    {"index": 2, "label": "hw", "color": (52, 152, 219)},
    {"index": 3, "label": "tt", "color": (155, 89, 182)},
)


def mse(y_true: Any, y_pred: Any) -> Any:
    """Custom loss retained for compatibility with the saved Keras model."""
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=-1)


@st.cache_resource(show_spinner=False)
def load_detection_model(model_path: str, model_version: int) -> Any:
    """Load and cache the model; model_version invalidates cache on replacement."""
    del model_version
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"mse": mse},
        compile=False,
    )


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode an uploaded image into OpenCV BGR format."""
    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def window_positions(image_size: int, window_size: int, step_size: int) -> list[int]:
    """Return scan positions, including the far edge of the image."""
    if image_size <= window_size:
        return [0]

    final_position = image_size - window_size
    positions = list(range(0, final_position + 1, step_size))
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def _update_detection(
    detection: dict[str, Any],
    score: float,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Extend a class box to cover every positive sliding-window patch."""
    detection["confidence"] = max(detection["confidence"], score)
    detection["hits"] += 1
    detection["x1"] = min(detection["x1"], x)
    detection["y1"] = min(detection["y1"], y)
    detection["x2"] = max(detection["x2"], x + width)
    detection["y2"] = max(detection["y2"], y + height)


def detect_objects(
    image: np.ndarray,
    model: Any,
    threshold: float = 0.60,
    window_size: int = 460,
    step_size: int = 50,
    batch_size: int = 32,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Locate one egg and choose its best class from model outputs 1-3."""
    image_height, image_width = image.shape[:2]
    patch_height = min(window_size, image_height)
    patch_width = min(window_size, image_width)
    y_positions = window_positions(image_height, patch_height, step_size)
    x_positions = window_positions(image_width, patch_width, step_size)

    detections: dict[int, dict[str, Any]] = {
        item["index"]: {
            **item,
            "confidence": 0.0,
            "hits": 0,
            "x1": image_width,
            "y1": image_height,
            "x2": 0,
            "y2": 0,
        }
        for item in CLASS_CONFIG
    }

    patches: list[np.ndarray] = []
    coordinates: list[tuple[int, int]] = []

    def predict_batch() -> None:
        if not patches:
            return

        model_input = np.asarray(patches, dtype=np.float32)
        if normalize:
            model_input /= 255.0

        predictions = np.asarray(model.predict(model_input, verbose=0))
        if predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]
        predictions = predictions.reshape(predictions.shape[0], -1)

        required_outputs = max(item["index"] for item in CLASS_CONFIG) + 1
        if predictions.shape[1] < required_outputs:
            raise ValueError(
                f"The model returned {predictions.shape[1]} output values per patch; "
                f"at least {required_outputs} are required for background plus classes 1-3."
            )

        for scores, (x, y) in zip(predictions, coordinates):
            for item in CLASS_CONFIG:
                score = float(scores[item["index"]])
                if score >= threshold:
                    _update_detection(
                        detections[item["index"]],
                        score,
                        x,
                        y,
                        patch_width,
                        patch_height,
                    )

        patches.clear()
        coordinates.clear()

    for y in y_positions:
        for x in x_positions:
            patch = image[y : y + patch_height, x : x + patch_width]
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
            patches.append(patch)
            coordinates.append((x, y))
            if len(patches) >= batch_size:
                predict_batch()
    predict_batch()

    positive_detections = [
        detection for detection in detections.values() if detection["hits"] > 0
    ]
    if not positive_detections:
        return []

    # The application contract is one egg per image. If more than one class
    # crosses the threshold, keep only the class with the strongest patch.
    best_detection = max(
        positive_detections, key=lambda detection: detection["confidence"]
    )
    return [best_detection]


def draw_detections(
    image: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    """Draw each detected class using a distinct color."""
    output = image.copy()
    image_height, image_width = output.shape[:2]
    font_scale = max(0.55, min(image_height, image_width) / 900)
    line_width = max(2, round(min(image_height, image_width) / 350))

    for detection in detections:
        x1 = max(0, int(detection["x1"]))
        y1 = max(0, int(detection["y1"]))
        x2 = min(image_width - 1, int(detection["x2"]))
        y2 = min(image_height - 1, int(detection["y2"]))
        color = detection["color"]
        label = f"{detection['label']}: {detection['confidence']:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            output,
            (x1, text_top),
            (min(image_width - 1, x1 + text_width + 10), y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 5, max(text_height + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )

    return output


def main() -> None:
    st.set_page_config(
        page_title="Object Detection",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Object Detection")
    st.caption(
        "Locate one egg and classify it as as, hw, or tt (model classes 1-3)."
    )

    with st.sidebar:
        st.header("Detection settings")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.60,
            step=0.05,
        )
        window_size = st.number_input(
            "Window size (pixels)", min_value=64, max_value=2048, value=460, step=16
        )
        step_size = st.number_input(
            "Step size (pixels)", min_value=10, max_value=512, value=50, step=10
        )
        normalize = st.checkbox(
            "Normalize pixels to 0-1",
            value=False,
            help="Leave off to match the preprocessing in the original code.",
        )
        st.divider()
        st.caption(f"Model: {DEFAULT_MODEL_PATH}")

    model = None
    model_error = None
    if DEFAULT_MODEL_PATH.is_file():
        try:
            with st.spinner("Loading detection model..."):
                model = load_detection_model(
                    str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_PATH.stat().st_mtime_ns
                )
        except Exception as exc:  # Surface model/configuration errors in the UI.
            model_error = str(exc)
    else:
        model_error = (
            f"Model file not found. Copy `sth_as-hw-tt.keras` into `{APP_DIR}` "
            "or set the `OBJDET_MODEL_PATH` environment variable."
        )

    if model_error:
        st.error(model_error)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="The image stays in memory while this page is open.",
    )

    if uploaded_file is None:
        st.info("Upload an image to start detection.")
        return

    try:
        image = decode_image(uploaded_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(original_rgb, caption="Uploaded image", use_container_width=True)

    if not st.button(
        "Detect objects", type="primary", disabled=model is None, use_container_width=True
    ):
        return

    try:
        with st.spinner("Scanning image patches..."):
            detections = detect_objects(
                image=image,
                model=model,
                threshold=confidence_threshold,
                window_size=int(window_size),
                step_size=int(step_size),
                normalize=normalize,
            )
            annotated = draw_detections(image, detections)
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        return

    st.subheader("Detection result")
    if detections:
        st.success(
            f"Detected one {detections[0]['label']} egg with "
            f"{detections[0]['confidence']:.2f} confidence."
        )
        result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Annotated image", use_container_width=True)

        summary = [
            {
                "Class": detection["label"],
                "Model output": detection["index"],
                "Confidence": round(detection["confidence"], 4),
                "Positive windows": detection["hits"],
            }
            for detection in detections
        ]
        st.dataframe(summary, hide_index=True, use_container_width=True)

        encoded, result_bytes = cv2.imencode(".jpg", annotated)
        if encoded:
            st.download_button(
                "Download annotated image",
                data=result_bytes.tobytes(),
                file_name=f"detected_{Path(uploaded_file.name).stem}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
    else:
        st.warning(
            f"No class exceeded the {confidence_threshold:.2f} confidence threshold."
        )


if __name__ == "__main__":
    main()
