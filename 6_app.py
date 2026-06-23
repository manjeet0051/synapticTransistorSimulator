"""
==============================================================
  IGZO/MgO Synaptic Transistor
  Real-World Object Recognition — Web App
==============================================================
  Run:  python app.py
  Open: http://localhost:5000

  REQUIRES:
    pip install flask tensorflow "numpy<2" pillow joblib scikit-learn
==============================================================
"""

import os
import io
import base64
import json
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "models_cnn_igzo")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_cnn_weights.h5")
RESULTS_PATH    = os.path.join(MODEL_DIR, "igzo_cnn_results.pkl")

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

CLASS_EMOJI = {
    'airplane'   : '✈',
    'automobile' : '🚗',
    'bird'       : '🐦',
    'cat'        : '🐱',
    'deer'       : '🦌',
    'dog'        : '🐶',
    'frog'       : '🐸',
    'horse'      : '🐴',
    'ship'       : '🚢',
    'truck'      : '🚛',
}

# CIFAR-10 normalization stats
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# ── Load models ───────────────────────────────────────────────
print("Loading IGZO CNN model...")

cnn_model        = None
feature_extractor = None
W_dense = b_dense = W_out = b_out = None
igzo_results     = {}

def load_models():
    global cnn_model, feature_extractor, W_dense, b_dense, W_out, b_out, igzo_results

    # ── Load IGZO dense weights ───────────────────────────────
    if os.path.exists(RESULTS_PATH):
        igzo_results = joblib.load(RESULTS_PATH)
        W_dense = igzo_results['W_dense']
        b_dense = igzo_results['b_dense']
        W_out   = igzo_results['W_out']
        b_out   = igzo_results['b_out']
        print(f"   IGZO dense weights loaded from: {RESULTS_PATH}")
    else:
        print(f"   WARNING: {RESULTS_PATH} not found.")
        print(f"   Run igzo_cnn_cifar10.py first to train the model.")

    # ── Load CNN backbone ─────────────────────────────────────
    backbone_path = os.path.join(MODEL_DIR, 'cnn_backbone.h5')
    if os.path.exists(backbone_path):
        cnn_model = keras.models.load_model(backbone_path)
        feature_extractor = Model(
            inputs  = cnn_model.input,
            outputs = cnn_model.get_layer('gap').output,
            name    = 'feature_extractor'
        )
        print(f"   CNN backbone loaded from: {backbone_path}")
    elif os.path.exists(CHECKPOINT_PATH):
        # Rebuild architecture and load weights
        from tensorflow.keras import layers
        def conv_block(x, filters):
            x = layers.Conv2D(filters, 3, padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, 3, padding='same',
                              kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2, 2)(x)
            x = layers.Dropout(0.3)(x)
            return x

        inp = layers.Input(shape=(32, 32, 3), name='image_input')
        x   = conv_block(inp, 32)
        x   = conv_block(x,   64)
        x   = conv_block(x,   128)
        x   = layers.Conv2D(128, 3, padding='same',
                            kernel_initializer='he_normal')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.Activation('relu')(x)
        x   = layers.GlobalAveragePooling2D(name='gap')(x)
        x   = layers.Dense(256, name='igzo_dense')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.Activation('relu')(x)
        x   = layers.Dropout(0.5)(x)
        out = layers.Dense(10, activation='softmax', name='output')(x)

        cnn_model = Model(inputs=inp, outputs=out, name='IGZO_CNN')
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
        cnn_model.load_weights(CHECKPOINT_PATH)
        feature_extractor = Model(
            inputs  = cnn_model.input,
            outputs = cnn_model.get_layer('gap').output,
            name    = 'feature_extractor'
        )
        print(f"   CNN weights loaded from: {CHECKPOINT_PATH}")
    else:
        print(f"   ERROR: No model files found in {MODEL_DIR}")
        print(f"   Run igzo_cnn_cifar10.py first!")

load_models()
print("Model ready!\n")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to 32x32, normalize exactly as during training."""
    img = img.convert('RGB')
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CIFAR_MEAN) / CIFAR_STD
    return arr[np.newaxis, ...]   # shape: (1, 32, 32, 3)


def igzo_forward(F, W1, b1, W2, b2):
    """IGZO dense layer forward pass (same as training script)."""
    Z1 = F @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    E  = np.exp(Z2)
    return E / E.sum(axis=1, keepdims=True)


def predict_image(img: Image.Image):
    """
    Full prediction pipeline:
    image → resize → normalize → CNN features → IGZO dense → probabilities
    """
    if feature_extractor is None or W_dense is None:
        return None, None

    # Preprocess
    x = preprocess_image(img)

    # CNN feature extraction (frozen backbone)
    features = feature_extractor.predict(x, verbose=0)   # (1, 128)

    # IGZO dense classification
    probs = igzo_forward(features, W_dense, b_dense, W_out, b_out)
    probs = probs[0]   # (10,)

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    # Build sorted results
    results = sorted(
        [{'class': CLASS_NAMES[i],
          'prob' : float(probs[i]) * 100,
          'emoji': CLASS_EMOJI[CLASS_NAMES[i]]}
         for i in range(10)],
        key=lambda x: x['prob'],
        reverse=True
    )
    return pred_name, results


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    model_ready = (feature_extractor is not None and W_dense is not None)
    acc = igzo_results.get('acc_final', igzo_results.get('acc_p1', 0))
    return render_template('index.html',
                           model_ready=model_ready,
                           accuracy=f"{acc*100:.2f}" if acc else "N/A",
                           class_names=CLASS_NAMES,
                           class_emoji=CLASS_EMOJI)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read uploaded image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Run prediction
        pred_name, results = predict_image(img)

        if pred_name is None:
            return jsonify({'error': 'Model not loaded. Run igzo_cnn_cifar10.py first.'}), 500

        # Convert image to base64 for display (original size, not resized)
        img_rgb = img.convert('RGB')
        # Also create the 32x32 version that was actually used
        img_32  = img_rgb.resize((32, 32), Image.LANCZOS)

        def to_b64(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'prediction' : pred_name,
            'emoji'      : CLASS_EMOJI[pred_name],
            'confidence' : results[0]['prob'],
            'results'    : results,
            'image_b64'  : to_b64(img_rgb),
            'image_32'   : to_b64(img_32),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    info = {
        'model_loaded' : feature_extractor is not None and W_dense is not None,
        'classes'      : CLASS_NAMES,
        'accuracy'     : igzo_results.get('acc_final', 0) * 100,
        'phase1_acc'   : igzo_results.get('acc_p1', 0) * 100,
        'avg_pot'      : float(igzo_results.get('avg_pot', 0)),
        'avg_dep'      : float(igzo_results.get('avg_dep', 0)),
        'pd_ratio'     : float(igzo_results.get('pd_ratio', 0)),
    }
    return jsonify(info)


if __name__ == '__main__':
    print("=" * 50)
    print("  IGZO Object Recognition App")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
