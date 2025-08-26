import sys
import os
import pandas as pd
from pathlib import Path

# Adiciona o path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "pipelines")))

# Importa as funções dos pipelines
from pipelines.csp import process_all_subjects as csp_all
from pipelines.csp_fractal import process_all_subjects as csp_frac_all
from pipelines.fbcsp import process_all_subjects as fbcsp_all
from pipelines.fbcsp_fractal import process_all_subjects as fbcsp_frac_all


def main():
    print("Iniciando processamento de todos os métodos para todos os sujeitos...")
    # Executa todos os pipelines e coleta as acurácias
    csp_accs = csp_all()
    csp_frac_accs = csp_frac_all()
    fbcsp_accs = fbcsp_all()
    fbcsp_frac_accs = fbcsp_frac_all()

    # Padroniza para sujeitos 1 a 9
    n_subjects = 9
    results_table = {
        "ID": list(range(1, n_subjects + 1)),
        "CSP": csp_accs[:n_subjects],
        "CSP-FRAC": csp_frac_accs[:n_subjects],
        "FBCSP": fbcsp_accs[:n_subjects],
        "FBCSP-FRAC": fbcsp_frac_accs[:n_subjects],
    }
    df_results = pd.DataFrame(results_table)
    means = {
        "ID": "mean",
        "CSP": df_results["CSP"].mean(),
        "CSP-FRAC": df_results["CSP-FRAC"].mean(),
        "FBCSP": df_results["FBCSP"].mean(),
        "FBCSP-FRAC": df_results["FBCSP-FRAC"].mean(),
    }
    df_means = pd.DataFrame([means])
    df_final = pd.concat([df_results, df_means], ignore_index=True)
    print(f"\n{'='*80}")
    print("TABELA FINAL DE RESULTADOS")
    print(f"{'='*80}")
    print(df_final.to_string(index=False, float_format="{:.4f}".format))
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "resultados_comparativos.csv"
    df_final.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nArquivo salvo em: {csv_path}")
    return df_final


if __name__ == "__main__":
    final_results = main()
