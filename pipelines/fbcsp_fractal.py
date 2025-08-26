import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bciflow.datasets.CBCIC import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.filterbank import filterbank
from bciflow.modules.sf.csp import csp
from methods.fractal import higuchi_fractal
from bciflow.modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import pandas as pd
from bciflow.modules.analysis.metric_functions import accuracy


def process_all_subjects():
    """Processa todos os 10 sujeitos do dataset e retorna as acurácias"""
    accuracies = []

    for subject in range(1, 11):  # Sujeitos de 1 a 10
        try:
            print(f"Processando sujeito {subject}...")

            # Carrega o dataset para o sujeito atual
            dataset = cbcic(subject=subject, path="dataset/wcci2020/")

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

            # Executa o kfold para o sujeito atual
            results = kfold(
                target=dataset,
                start_window=dataset["events"]["cue"][0] + 0.5,
                pre_folding=pre_folding,
                pos_folding=pos_folding,
            )

            # Calcula a acurácia
            df = pd.DataFrame(results)
            acc = accuracy(df)
            accuracies.append(acc)

            print(f"Sujeito {subject}: Accuracy = {acc:.4f}")

        except Exception as e:
            print(f"Erro ao processar sujeito {subject}: {e}")
            accuracies.append(None)

    return accuracies


# Executa a função principal
if __name__ == "__main__":
    print("Iniciando processamento para todos os 10 sujeitos...")
    all_accuracies = process_all_subjects()

    print("\n" + "=" * 50)
    print("RESUMO DAS ACURÁCIAS:")
    print("=" * 50)

    for i, acc in enumerate(all_accuracies, 1):
        if acc is not None:
            print(f"Sujeito {i}: {acc:.4f}")
        else:
            print(f"Sujeito {i}: Erro no processamento")

    # Calcula estatísticas
    valid_accuracies = [acc for acc in all_accuracies if acc is not None]
    if valid_accuracies:
        print(
            f"\nMédia das acurácias: {sum(valid_accuracies)/len(valid_accuracies):.4f}"
        )
        print(f"Maior acurácia: {max(valid_accuracies):.4f}")
        print(f"Menor acurácia: {min(valid_accuracies):.4f}")
