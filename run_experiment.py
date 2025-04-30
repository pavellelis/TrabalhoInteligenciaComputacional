# run_experiment.py
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import sem # Para Standard Error of the Mean
# interp1d é útil para alinhar históricos de runs com FEs diferentes
from scipy.interpolate import interp1d
import pickle # Opcional: para salvar/carregar resultados detalhados

# Adiciona diretório atual ao path para garantir que os módulos sejam encontrados
# (pode não ser necessário dependendo de como você executa)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa a função principal que executa as runs
try:
    from aspsogmpb_runner import main_aspsogmpb
except ImportError as e:
    print(f"Erro fatal: Não foi possível importar 'main_aspsogmpb' de 'aspsogmpb_runner.py'.")
    print(f"Detalhe: {e}")
    print("Certifique-se de que todos os arquivos .py estão no mesmo diretório.")
    exit()

# ==================================================
# Função Auxiliar para Processar e Plotar Históricos
# ==================================================
def process_and_plot_histories(run_results, max_evals_total, run_number):
    """Processa históricos de múltiplas runs e gera gráficos."""

    print("\nProcessando dados históricos para gráficos...")

    # --- Prepara eixo X comum (FEs) ---
    # Cria um eixo de FEs de 1 até max_evals_total
    common_fe_axis = np.arange(1, max_evals_total + 1)
    all_runs_data_interp = {} # Dicionário para guardar dados interpolados por métrica

    # Métricas a serem processadas e seus nomes para os gráficos
    metrics_to_process = {
        "pop_size_history": "Tamanho da População",
        "species_count_history": "Número de Espécies (Total)",
        "active_species_count_history": "Número de Espécies (Ativas)",
        "deactivation_history": "Raio de Desativação (ra)",
        "shift_severity_history": "Shift Severity Estimado (ŝ)"
    }
    history_data_available = False # Flag para saber se algum histórico foi coletado

    # --- Interpola dados de cada run para o eixo comum ---
    print("Interpolando dados históricos...")
    fe_history_list = run_results.get("fe_history", [])
    if not fe_history_list:
        print("Aviso: Histórico de FEs ('fe_history') não encontrado nos resultados.")
        return # Não pode processar sem FEs

    for metric_key, _ in metrics_to_process.items():
        interpolated_runs = np.full((run_number, max_evals_total), np.nan) # Inicializa com NaN
        history_list = run_results.get(metric_key, [])

        if not history_list or len(history_list) != run_number or len(fe_history_list) != run_number:
             print(f"Aviso: Dados históricos ausentes ou de tamanho incorreto para '{metric_key}'. Pulando.")
             continue

        history_data_available = True # Pelo menos um histórico está presente
        for i in range(run_number):
            run_fes = fe_history_list[i]
            run_values = history_list[i]

            # Só interpola se tivermos pelo menos 2 pontos de dados para a run
            if len(run_fes) > 1 and len(run_values) == len(run_fes):
                # Cria função de interpolação linear para esta run
                # bounds_error=False e fill_value=(run_values[0], run_values[-1])
                # preenchem fora dos limites com o primeiro/último valor
                # (evita extrapolação linear que pode dar valores irreais)
                interp_func = interp1d(run_fes, run_values, kind='linear', bounds_error=False,
                                       fill_value=(run_values[0], run_values[-1]))
                # Aplica no eixo comum
                interpolated_runs[i, :] = interp_func(common_fe_axis)
            elif len(run_fes) == 1: # Apenas um ponto de dado
                 # Encontra o índice correspondente ao FE e preenche a partir dali
                 fe_idx = int(run_fes[0]) - 1
                 if 0 <= fe_idx < max_evals_total:
                      interpolated_runs[i, fe_idx:] = run_values[0] # Preenche do ponto em diante
            # Se run_fes estiver vazio, a linha permanece NaN

        all_runs_data_interp[metric_key] = interpolated_runs

    if not history_data_available:
        print("Nenhum dado histórico válido encontrado para plotar.")
        return

    # --- Calcula Médias e SEM ---
    processed_metrics = {}
    print("Calculando médias e erro padrão...")
    for metric_key, data_array in all_runs_data_interp.items():
        mean_vals = np.nanmean(data_array, axis=0)
        # Calcula SEM apenas onde há dados suficientes (mais de 1 run contribuiu)
        valid_counts_per_fe = np.sum(~np.isnan(data_array), axis=0)
        sem_vals = np.full_like(mean_vals, np.nan) # Inicia com NaN
        # Calcula SEM apenas para pontos com pelo menos 2 dados não-NaN
        indices_calc_sem = np.where(valid_counts_per_fe >= 2)[0]
        if indices_calc_sem.size > 0:
            sem_vals[indices_calc_sem] = sem(data_array[:, indices_calc_sem], axis=0, nan_policy='omit')

        processed_metrics[metric_key] = {'mean': mean_vals, 'sem': sem_vals}

    # --- Gera Gráficos ---
    print("Gerando gráficos...")

    # 1. Gráfico de Desempenho (Erro Corrente)
    plt.style.use('seaborn-v0_8-darkgrid') # Estilo um pouco mais moderno
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    current_error_all_runs = run_results["current_error"]
    mean_current_error = np.nanmean(current_error_all_runs, axis=0)
    valid_error_counts = np.sum(~np.isnan(current_error_all_runs), axis=0)
    sem_current_error = np.full_like(mean_current_error, np.nan)
    indices_calc_sem_err = np.where(valid_error_counts >= 2)[0]
    if indices_calc_sem_err.size > 0:
        sem_current_error[indices_calc_sem_err] = sem(current_error_all_runs[:, indices_calc_sem_err], axis=0, nan_policy='omit')
    evaluations_perf = np.arange(1, len(mean_current_error) + 1)

    ax1.plot(evaluations_perf, mean_current_error, label='Erro Médio Corrente ($E_C$)', linewidth=1.5, color='royalblue')
    # Preenche apenas onde SEM é válido
    valid_sem_indices_err = ~np.isnan(sem_current_error)
    ax1.fill_between(evaluations_perf[valid_sem_indices_err],
                     (mean_current_error - sem_current_error)[valid_sem_indices_err],
                     (mean_current_error + sem_current_error)[valid_sem_indices_err],
                     color='lightblue', alpha=0.4, label='Erro Padrão da Média (SEM)')
    ax1.set_xlabel('Avaliações de Fitness (FEs)')
    ax1.set_ylabel('Erro Médio Corrente')
    ax1.set_title(f'Desempenho Médio do SPSO+AP+AD ({run_number} Runs)')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    ax1.legend()
    ax1.set_xlim(1, max_evals_total)
    min_y_err = np.nanmin(mean_current_error[np.isfinite(mean_current_error)]) if np.any(np.isfinite(mean_current_error)) else 1e-6
    ax1.set_ylim(bottom=max(min_y_err / 5, 1e-7)) # Ajuste fino do limite inferior
    fig1.tight_layout()
    fig1.savefig("performance_spsopapad.png", dpi=300)
    print("Gráfico de desempenho salvo como performance_spsopapad.png")
    plt.close(fig1) # Fecha a figura para liberar memória

    # 2. Gráfico de Dinâmica Populacional
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Compartilha eixo X
    if "pop_size_history" in processed_metrics:
        mean_pop = processed_metrics["pop_size_history"]['mean']
        sem_pop = processed_metrics["pop_size_history"]['sem']
        valid_sem_pop = ~np.isnan(sem_pop)
        ax2a.plot(common_fe_axis, mean_pop, label='Tamanho Médio da População', color='forestgreen')
        ax2a.fill_between(common_fe_axis[valid_sem_pop], (mean_pop - sem_pop)[valid_sem_pop], (mean_pop + sem_pop)[valid_sem_pop], color='lightgreen', alpha=0.4)
        ax2a.set_ylabel('Número de Partículas')
        ax2a.set_title('Dinâmica Populacional e de Espécies Média do SPSO+AP+AD')
        ax2a.grid(True, which="both", ls="--", linewidth=0.5)
        ax2a.legend()
        ax2a.set_xlim(1, max_evals_total)

    if "species_count_history" in processed_metrics and "active_species_count_history" in processed_metrics:
        mean_spec = processed_metrics["species_count_history"]['mean']
        sem_spec = processed_metrics["species_count_history"]['sem']
        valid_sem_spec = ~np.isnan(sem_spec)
        mean_active = processed_metrics["active_species_count_history"]['mean']
        sem_active = processed_metrics["active_species_count_history"]['sem']
        valid_sem_active = ~np.isnan(sem_active)

        ax2b.plot(common_fe_axis, mean_spec, label='Espécies (Total)', color='darkorange')
        ax2b.fill_between(common_fe_axis[valid_sem_spec], (mean_spec - sem_spec)[valid_sem_spec], (mean_spec + sem_spec)[valid_sem_spec], color='moccasin', alpha=0.4)
        ax2b.plot(common_fe_axis, mean_active, label='Espécies (Ativas)', linestyle='--', color='firebrick')
        ax2b.fill_between(common_fe_axis[valid_sem_active], (mean_active - sem_active)[valid_sem_active], (mean_active + sem_active)[valid_sem_active], color='lightcoral', alpha=0.4)
        ax2b.set_xlabel('Avaliações de Fitness (FEs)')
        ax2b.set_ylabel('Número de Espécies')
        ax2b.grid(True, which="both", ls="--", linewidth=0.5)
        ax2b.legend()
        ax2b.set_xlim(1, max_evals_total)

    fig2.tight_layout()
    fig2.savefig("population_dynamics_spsopapad.png", dpi=300)
    print("Gráfico de dinâmica populacional salvo como population_dynamics_spsopapad.png")
    plt.close(fig2)

    # 3. Gráfico de Parâmetros Adaptativos (Ex: Raio de Desativação e Shift Severity)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3b = ax3.twinx() # Eixo Y secundário para Shift Severity
    plot_lines = [] # Para juntar legendas

    if "deactivation_history" in processed_metrics:
        mean_deact = processed_metrics["deactivation_history"]['mean']
        sem_deact = processed_metrics["deactivation_history"]['sem']
        valid_sem_deact = ~np.isnan(sem_deact)
        line1, = ax3.plot(common_fe_axis, mean_deact, label='$r_a$ Médio (CurrentDeactivation)', color='purple')
        ax3.fill_between(common_fe_axis[valid_sem_deact], (mean_deact - sem_deact)[valid_sem_deact], (mean_deact + sem_deact)[valid_sem_deact], color='plum', alpha=0.4)
        ax3.set_xlabel('Avaliações de Fitness (FEs)')
        ax3.set_ylabel('Valor do Raio de Desativação ($r_a$)', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        plot_lines.append(line1)

    if "shift_severity_history" in processed_metrics:
        mean_shift = processed_metrics["shift_severity_history"]['mean']
        sem_shift = processed_metrics["shift_severity_history"]['sem']
        valid_sem_shift = ~np.isnan(sem_shift)
        line2, = ax3b.plot(common_fe_axis, mean_shift, label='Shift Severity Médio (ŝ)', color='teal', linestyle=':')
        ax3b.fill_between(common_fe_axis[valid_sem_shift], (mean_shift - sem_shift)[valid_sem_shift], (mean_shift + sem_shift)[valid_sem_shift], color='paleturquoise', alpha=0.4)
        ax3b.set_ylabel('Valor do Shift Severity (ŝ)', color='teal')
        ax3b.tick_params(axis='y', labelcolor='teal')
        plot_lines.append(line2)

    if plot_lines: # Só adiciona legenda se algo foi plotado
         ax3.legend(plot_lines, [l.get_label() for l in plot_lines], loc='best')

    ax3.set_title('Adaptação Média de Parâmetros ($r_a$ e ŝ)')
    ax3.grid(True, which="both", ls="--", linewidth=0.5, axis='x') # Grid só no X para não poluir
    ax3.set_xlim(1, max_evals_total)
    fig3.tight_layout()
    fig3.savefig("adaptive_params_spsopapad.png", dpi=300)
    print("Gráfico de parâmetros adaptativos salvo como adaptive_params_spsopapad.png")
    plt.close(fig3)

    print("Processamento de gráficos concluído.")


# ==================================================
# Script Principal para Rodar o Experimento ASPSO+GMPB
# ==================================================
if __name__ == "__main__":
    start_time_script = time.time() # Tempo total do script
    print("**********************************************************************")
    print("* ASPSO com Tamanho Populacional Adaptativo e Desativação (ASPSO+AP+AD) *")
    print("* Teste no Generalized Moving Peaks Benchmark (GMPB)         *")
    print("**********************************************************************")
    print("(Implementação Python baseada no código MATLAB de Delaram Yazdani et al.)\n")

    # -------- Parâmetros do Benchmark e Execução --------
    # Use valores menores para teste inicial, depois ajuste para os do artigo/Main.m
    run_number = 5          # Original é 31 (Reduzido)
    peak_number = 10
    change_frequency = 500  # Original é 5000 (Reduzido)
    dimension = 5           # Original é 10 (Reduzido)
    shift_severity = 1.0
    environment_number = 10 # Original é 100 (Reduzido)

    # Calcula o total de FEs esperado
    max_evals_total = change_frequency * environment_number

    print("Parâmetros do Experimento:")
    print(f"- Número de Runs: {run_number}")
    print(f"- Número de Picos: {peak_number}")
    print(f"- Frequência de Mudança (FEs): {change_frequency}")
    print(f"- Dimensão: {dimension}")
    print(f"- Severidade do Shift: {shift_severity}")
    print(f"- Número de Ambientes: {environment_number}")
    print(f"- Total de Avaliações por Run: {max_evals_total}")
    print("-" * 60)

    # -------- Executa o Experimento --------
    start_time_exec = time.time()

    try:
        # Chama a função principal que executa todas as runs e retorna os dados
        run_results = main_aspsogmpb(
            peak_number, change_frequency, dimension, shift_severity,
            environment_number, run_number
        )

        end_time_exec = time.time()
        print("-" * 60)
        print(f"Tempo de Execução (Simulação): {end_time_exec - start_time_exec:.2f} segundos.")
        print("-" * 60)

        # -------- Processa e Exibe Resultados --------
        offline_error = run_results.get("offline_error") # Usa .get para segurança

        if offline_error is not None and not np.all(np.isnan(offline_error)):
            mean_offline_error = np.nanmean(offline_error)
            std_offline_error = np.nanstd(offline_error)
            valid_runs = np.count_nonzero(~np.isnan(offline_error))
            if valid_runs >= 2: # SEM só faz sentido com N >= 2
                 sem_offline_error = std_offline_error / np.sqrt(valid_runs)
            elif valid_runs == 1:
                 sem_offline_error = 0.0 # Ou NaN? SEM de 1 ponto é 0 ou indefinido.
            else:
                 sem_offline_error = np.nan

            print("Resultado Final (Erro Offline):")
            print(f"- Média: {mean_offline_error:.6e}")
            if not np.isnan(sem_offline_error):
                 print(f"- Erro Padrão (SEM): {sem_offline_error:.6e}")
            else:
                 print("- Erro Padrão (SEM): N/A (runs insuficientes/inválidas)")
            print(f"(Baseado em {valid_runs} runs válidas de {run_number})")

            # --- Chama a função para gerar e salvar os gráficos ---
            if valid_runs > 0: # Só plota se houver dados válidos
                process_and_plot_histories(run_results, max_evals_total, run_number)
            else:
                print("\nNão foi possível gerar gráficos: nenhuma run válida completou.")


            # Opcional: Salvar todos os dados coletados para análise posterior
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f'run_results_spsopapad_{timestamp}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(run_results, f)
                print(f"\nResultados detalhados salvos em: {filename}")
            except Exception as pkl_e:
                print(f"\nAviso: Falha ao salvar resultados detalhados em pickle: {pkl_e}")


        else:
            print("Não foi possível calcular o erro offline (nenhuma run válida ou erro durante a execução).")


    except Exception as e:
        print("\n!!! Ocorreu um erro fatal durante a execução do experimento !!!")
        import traceback
        traceback.print_exc()
        print(f"Erro: {e}")

    end_time_script = time.time()
    print("-" * 60)
    print(f"Tempo total do script: {end_time_script - start_time_script:.2f} segundos.")
    print("Fim do script.")