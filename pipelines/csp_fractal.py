import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bciflow.datasets.CBCIC import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.sf.csp import csp
from methods.fractal import higuchi_fractal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import pandas as pd
from bciflow.modules.analysis.metric_functions import accuracy


def process_all_subjects():
    accuracies = []

    for subject in range(1, 10):
        try:
            print(f"Processando sujeito {subject}...")

            dataset = cbcic(subject=subject, path="dataset/wcci2020/")

            pre_folding = {"tf": (chebyshevII, {})}

            sf = csp()
            fe = higuchi_fractal
            clf = lda()

            pos_folding = {
                "sf": (sf, {}),
                "fe": (fe, {"flating": True}),
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
            accuracies.append(acc)

            print(f"Sujeito {subject}: Accuracy = {acc:.4f}")

        except Exception as e:
            print(f"Erro ao processar sujeito {subject}: {e}")
            accuracies.append(None)

    return accuracies


if __name__ == "__main__":
    print("Iniciando processamento para todos os 9 sujeitos...")
    all_accuracies = process_all_subjects()

    print("\n" + "=" * 50)
    print("RESUMO DAS ACURÁCIAS:")
    print("=" * 50)

    for i, acc in enumerate(all_accuracies, 1):
        if acc is not None:
            print(f"Sujeito {i}: {acc:.4f}")
        else:
            print(f"Sujeito {i}: Erro no processamento")

    valid_accuracies = [acc for acc in all_accuracies if acc is not None]
    if valid_accuracies:
        print(
            f"\nMédia das acurácias: {sum(valid_accuracies)/len(valid_accuracies):.4f}"
        )
        print(f"Maior acurácia: {max(valid_accuracies):.4f}")
        print(f"Menor acurácia: {min(valid_accuracies):.4f}")
