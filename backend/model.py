import numpy as np
import tensorflow as tf
import os

_models = None


def get_models():
    global _models

    if _models is None:

        base_dir = os.path.dirname(__file__)

        configs = [
            ("EfficientNetB0", "model_EfficientNetB0_best.keras"),
            ("ResNet50", "model_ResNet50_best.keras"),
            ("DenseNet121", "model_DenseNet121_best.keras"),
        ]

        _models = []

        dummy = np.zeros((1,128,128,3), dtype=np.float32)

        for name, filename in configs:

            path = os.path.join(base_dir, filename)

            print(f"⚙️ Loading {name}...")

            m = tf.keras.models.load_model(path)

            m.predict(dummy, verbose=0)

            _models.append((name, m))

            print(f"✅ {name} loaded successfully")

    return _models


def predict_age(volume):

    models = get_models()

    preds = []

    for name, m in models:

        p = float(
            m.predict(volume.astype(np.float32), verbose=0)
            .flatten()[0]
        )

        print(f"{name}: {p:.1f}")

        preds.append(p)

    ensemble = float(np.mean(preds))

    print(f"Ensemble: {ensemble:.1f}")

    return ensemble