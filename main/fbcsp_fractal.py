from bciflow.datasets.CBCIC import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.filterbank import filterbank
from bciflow.modules.sf.csp import csp
from methods.fractal import higuchi_fractal
from bciflow.modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import pandas as pd
from bciflow.modules.analysis.metric_functions import accuracy

dataset = cbcic(subject=1, path="dataset/wcci2020/")

pre_folding = {"tf": (filterbank, {"kind_bp": "chebyshevII"})}

sf = csp()
fe = higuchi_fractal
fs = MIBIF(8, clf=lda())
clf = lda()

pos_folding = {
    "sf": (sf, {}),
    "fe": (fe, {}),
    "fs": (fs, {}),
    "clf": (clf, {}),
}

results = kfold(
    target=dataset,
    start_window=dataset["events"]["cue"][0] + 0.5,
    pre_folding=pre_folding,
    pos_folding=pos_folding,
)

df = pd.DataFrame(results)
acc = accuracy(df)
print(f"Accuracy: {acc:.4f}")
