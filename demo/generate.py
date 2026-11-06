"""Build a standalone presentation demo from real Fashion-MNIST predictions."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import get_stratified_split
from src.models import get_rbf_svc_pipeline

NAMES = ['Футболка / топ', 'Штани', 'Пуловер', 'Сукня', 'Пальто', 'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики']

def main():
    (train_x, train_y), (test_x, test_y) = get_stratified_split(cache_dir=str(ROOT / 'data/fashion_mnist'))
    rng = np.random.default_rng(42)
    indices = np.concatenate([rng.choice(np.flatnonzero(test_y == label), size=12, replace=False) for label in range(10)])
    model = get_rbf_svc_pipeline(n_components=50, C=5.0, gamma='scale', random_state=42)
    # Enable libsvm's internal probability calibration; the project pipeline
    # defaults to probability=False and does not itself expose probabilities.
    model.set_params(clf__probability=True)
    print('Training RBF-SVC and calibrating probabilities on 12,000 samples...', flush=True)
    model.fit(train_x, train_y)
    print('Predicting 120 held-out examples...', flush=True)
    predictions = model.predict(test_x[indices])
    probabilities = model.predict_proba(test_x[indices])
    classes = model.named_steps['clf'].classes_
    samples = []
    for index, prediction, probability in zip(indices, predictions, probabilities):
        img = Image.fromarray(test_x[index].reshape(28, 28).astype('uint8'), mode='L')
        out = BytesIO()
        img.save(out, format='PNG')
        probs_by_class = {int(label): float(value) for label, value in zip(classes, probability)}
        samples.append({'id': int(index), 'truth': int(test_y[index]), 'prediction': int(prediction), 'probabilities': [round(probs_by_class[i], 6) for i in range(len(NAMES))], 'image': base64.b64encode(out.getvalue()).decode('ascii')})
    template = (ROOT / 'demo/template.html').read_text(encoding='utf-8')
    output = template.replace('/* DATA_PLACEHOLDER */', 'const samples = ' + json.dumps(samples, ensure_ascii=False) + ';\nconst names = ' + json.dumps(NAMES, ensure_ascii=False) + ';')
    (ROOT / 'demo/demo.html').write_text(output, encoding='utf-8')
    print(f"Saved demo/demo.html: {len(samples)} examples, {sum(s['truth'] != s['prediction'] for s in samples)} errors")

if __name__ == '__main__':
    main()
