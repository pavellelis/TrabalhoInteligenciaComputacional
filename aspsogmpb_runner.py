# aspsogmpb_runner.py
import numpy as np
import time
import os
# Importa as funções dos outros módulos
try:
    from gmpb_benchmark import benchmark_generator_aspsogmpb
    # Importa ambas funções de utils
    from aspsogmpb_utils import initializing_optimizer_aspsogmpb, creating_species
    # Importa ambas funções do core
    from aspsogmpb_core import optimization_aspsogmpb, reaction_aspsogmpb
    # Importa fitness, pois é usado na inicialização
    from fitness_eval import fitness_aspsogmpb
except ImportError as e:
    print(f"Erro fatal ao importar módulos necessários em aspsogmpb_runner: {e}")
    print("Certifique-se que os arquivos .py anteriores estão completos e no mesmo diretório.")
    exit()

# =============================================
# Função Principal de Execução (Orquestrador)
# =============================================
def main_aspsogmpb(peak_number, change_frequency, dimension, shift_severity,
                   environment_number, run_number):
    """
    Executa múltiplas runs do ASPSO no GMPB, orquestrando a inicialização,
    o loop de otimização/reação e a coleta de dados para gráficos.
    Equivalente a main_ASPSO_GMPB.m.

    Args:
        peak_number (int): Número de picos.
        change_frequency (int): Avaliações por ambiente.
        dimension (int): Dimensionalidade.
        shift_severity (float): Severidade da mudança.
        environment_number (int): Número de ambientes.
        run_number (int): Número de execuções independentes.

    Returns:
        dict: Dicionário contendo resultados agregados e históricos:
            "problem_final": Estado final do problema da última run.
            "current_error": Array (Runs x MaxEvals) de erro corrente.
            "offline_error": Array (Runs,) de erro offline médio por run.
            "fe_history": Lista de listas [run][step] de contagem de FEs.
            "pop_size_history": Lista de listas [run][step] de tamanho da população.
            "species_count_history": Lista de listas [run][step] de número de espécies.
            "active_species_count_history": Lista de listas [run][step] de espécies ativas.
            "deactivation_history": Lista de listas [run][step] do valor de CurrentDeactivation.
            "shift_severity_history": Lista de listas [run][step] do valor estimado de ShiftSeverity.
    """
    max_evals_total = change_frequency * environment_number
    if max_evals_total <= 0:
        raise ValueError("max_evals_total deve ser positivo. Verifique change_frequency e environment_number.")

    # --- Inicializa estruturas para guardar resultados de todas as runs ---
    offline_error_all_runs = np.full(run_number, np.nan)
    # Usa max_evals_total, mesmo que algumas runs possam ter FE ligeiramente maior se pararem tarde
    current_error_all_runs = np.full((run_number, max_evals_total), np.nan)
    problem_final = None # Guarda o estado da última run

    # Listas para guardar históricos de TODAS as runs
    all_fe_histories = []
    all_pop_size_histories = []
    all_species_count_histories = []
    all_active_species_count_histories = []
    all_deactivation_histories = []
    all_shift_severity_histories = [] # Novo: histórico do shift severity estimado


    start_time_total = time.time()
    print(f"Iniciando simulação ASPSO+AP+AD para {run_number} runs...")
    print(f"Total de avaliações por run esperado: {max_evals_total}")

    # --- Loop Principal das Runs ---
    for run_counter in range(run_number):
        run_start_time = time.time()
        print(f"\n--- Iniciando Run {run_counter + 1} / {run_number} ---")

        # Define semente para GERAÇÃO DO PROBLEMA
        np.random.seed(run_counter + 1)
        print(f"Gerando problema com semente {run_counter + 1}...")
        try:
            problem = benchmark_generator_aspsogmpb(peak_number, change_frequency, dimension,
                                                    shift_severity, environment_number)
        except Exception as e:
             print(f"Erro fatal ao gerar benchmark na run {run_counter + 1}: {e}")
             # Preenche erros com NaN e pula para próxima run? Ou para? Vamos parar.
             raise RuntimeError(f"Falha ao gerar benchmark na run {run_counter + 1}") from e


        # Define semente aleatória para o OTIMIZADOR
        np.random.seed()
        print("Semente do otimizador definida como aleatória.")

        # --- Inicialização do Dicionário 'optimizer' ---
        print("Configurando parâmetros do otimizador...")
        optimizer = {}
        try:
            # Parâmetros fixos (Tabela 3 do artigo)
            optimizer['SwarmMember'] = 5           # n (tamanho da espécie)
            optimizer['NewlyAddedPopulationSize'] = 5 # m (indivíduos adicionados)
            optimizer['x'] = 0.729843788         # Coeficiente PSO
            optimizer['c1'] = 2.05
            optimizer['c2'] = 2.05
            optimizer['rho'] = 0.7               # Fator para MaxDeactivation
            optimizer['mu'] = 0.2                # Fator para MinDeactivation
            optimizer['gama'] = 0.1               # Fator de decaimento de beta
            optimizer['Nmax'] = 30                # Limite de espécies para anti-convergência
            optimizer['InitialPopulationSize'] = 50 # Tamanho inicial

            # Parâmetros dependentes do problema
            optimizer['Dimension'] = problem['Dimension']
            optimizer['MaxCoordinate'] = problem['MaxCoordinate']
            optimizer['MinCoordinate'] = problem['MinCoordinate']

            # Parâmetros adaptativos (valores iniciais)
            optimizer['ShiftSeverity'] = 1.0       # Estimativa inicial do shift (será atualizada)
            optimizer['beta'] = 1.0                # Fator de decaimento (resetado a cada mudança)
            optimizer['teta'] = optimizer['ShiftSeverity'] # Limiar Tracker = Shift inicial
            optimizer['MaxDeactivation'] = optimizer['rho'] * optimizer['ShiftSeverity'] # r_a^max inicial
            optimizer['MinDeactivation'] = optimizer['mu'] * np.sqrt(max(optimizer['Dimension'], 0)) # r_a^min
            optimizer['CurrentDeactivation'] = optimizer['MaxDeactivation'] # r_a inicial

            # Calcula raios iniciais baseados em N=PeakNumber (aproximação inicial)
            initial_num_species_guess = max(1, problem['PeakNumber']) # Evita N=0
            range_coord_init = optimizer['MaxCoordinate'] - optimizer['MinCoordinate']
            dim_eff = max(optimizer['Dimension'], 1e-9) # Evita D=0
            n_pow_inv_d_init = initial_num_species_guess ** (1.0 / dim_eff)
            d_boa_init = range_coord_init / max(n_pow_inv_d_init, 1e-9)
            optimizer['ExclusionLimit'] = 0.5 * d_boa_init # r_excl inicial
            optimizer['GenerateRadious'] = 0.3 * d_boa_init # r_generate inicial

            # Estruturas de dados do otimizador
            optimizer['particle'] = [] # Lista para guardar as partículas (dicionários)
            optimizer['tracker'] = []  # Lista de índices das espécies tracker (na species_list atual)

            # Ajusta tamanho inicial da população
            optimizer['InitialPopulationSize'] = max(optimizer['InitialPopulationSize'], optimizer['SwarmMember'])

        except KeyError as e:
             print(f"Erro: Chave esperada não encontrada no dicionário 'problem' ao inicializar 'optimizer': {e}")
             raise RuntimeError(f"Falha ao inicializar otimizador na run {run_counter + 1}")

        # --- Criação da População Inicial ---
        print(f"Inicializando população com {optimizer['InitialPopulationSize']} partículas...")
        # Reseta contadores e flags do problema para esta run
        problem['FE'] = 0
        problem['RecentChange'] = 0
        problem['Environmentcounter'] = 1
        problem['CurrentError'] = np.full(max_evals_total, np.nan) # Reseta array de erro

        try:
            for _ in range(optimizer['InitialPopulationSize']):
                particle_state, problem = initializing_optimizer_aspsogmpb(
                    optimizer['MinCoordinate'], optimizer['MaxCoordinate'],
                    optimizer['Dimension'], problem)
                optimizer['particle'].append(particle_state)
                # Verifica se a inicialização causou mudança (improvável aqui)
                if problem.get('RecentChange', 0):
                    print("AVISO: Mudança detectada durante a inicialização da população!")
                    problem['RecentChange'] = 0 # Reseta a flag para permitir o início do loop
        except Exception as e:
            print(f"Erro durante a inicialização da população na run {run_counter + 1}: {e}")
            raise RuntimeError(f"Falha ao inicializar população na run {run_counter + 1}") from e


        print(f"População inicial criada. FEs usados: {problem['FE']}")

        # --- Listas para histórico DESTA run ---
        pop_size_history_run = []
        species_count_history_run = []
        active_species_count_history_run = []
        deactivation_history_run = []
        shift_severity_history_run = [] # Log do Shift Severity
        fe_history_run = []
        last_logged_fe = -1 # Para evitar logs duplicados no mesmo FE
        species_list_current = [] # Guarda as espécies da iteração (inicialmente vazia)

        # --- Loop Principal de Otimização (Iterações / FEs) ---
        iteration = 0
        max_evals_run = problem['MaxEvals'] # Pega o total de FEs para esta run

        while problem['FE'] < max_evals_run:
            iteration += 1
            fe_before_iter = problem['FE']

            # --- Passo de Otimização ---
            try:
                # Guarda as espécies antes da otimização para passar para reaction se necessário
                # No entanto, a lógica do código core já lida com recriação interna.
                # Passar a lista atualizada por optimization para reaction é o comportamento
                # inferido do MATLAB e do código core.
                optimizer, problem, species_list_current = optimization_aspsogmpb(optimizer, problem)
            except Exception as e:
                 print(f"\nERRO FATAL na função optimization_aspsogmpb na run {run_counter+1}, iteração {iteration}, FE={problem['FE']}:")
                 import traceback
                 traceback.print_exc()
                 # Decide se para a run ou o programa inteiro
                 # Vamos parar a run e preencher com NaN
                 offline_error_all_runs[run_counter] = np.nan
                 # Não adiciona históricos desta run
                 break # Sai do loop while

            # --- Logging de Dados ---
            current_fe = problem['FE']
            # Log apenas se FE avançou e está dentro do limite
            if current_fe > last_logged_fe and current_fe <= max_evals_run:
                try:
                    pop_size = len(optimizer.get('particle', []))
                    species_count = len(species_list_current) # Usa a lista retornada por optimization
                    active_count = sum(1 for s in species_list_current if s.get('Active', 1))
                    deactivation_val = optimizer.get('CurrentDeactivation', np.nan)
                    shift_sev_val = optimizer.get('ShiftSeverity', np.nan) # Loga o shift severity atual

                    pop_size_history_run.append(pop_size)
                    species_count_history_run.append(species_count)
                    active_species_count_history_run.append(active_count)
                    deactivation_history_run.append(deactivation_val)
                    shift_severity_history_run.append(shift_sev_val) # Adiciona log
                    fe_history_run.append(current_fe)
                    last_logged_fe = current_fe
                except Exception as log_e:
                     print(f"Aviso: Erro durante o logging na iteração {iteration}, FE={current_fe}: {log_e}")


            # --- Reação à Mudança ---
            if problem.get('RecentChange', 0):
                fe_before_reaction = problem['FE']
                print(f"\nINFO: Detectada mudança para ambiente {problem['Environmentcounter']} na Iter {iteration} (FE={fe_before_reaction})")
                try:
                    # Passa as espécies atuais (retornadas por optimization) para reaction
                    optimizer, problem = reaction_aspsogmpb(optimizer, problem, species_list_current)
                    problem['RecentChange'] = 0 # Reseta a flag após a reação bem-sucedida
                except Exception as e:
                     print(f"\nERRO FATAL na função reaction_aspsogmpb na run {run_counter+1}, iteração {iteration}, FE={problem['FE']}:")
                     import traceback
                     traceback.print_exc()
                     offline_error_all_runs[run_counter] = np.nan
                     break # Sai do loop while

                # --- Log de Dados (APÓS a reação) ---
                # Logar o estado atualizado, especialmente os parâmetros resetados pela reação
                current_fe_after_reaction = problem['FE']
                if current_fe_after_reaction > last_logged_fe and current_fe_after_reaction <= max_evals_run:
                     try:
                        # A contagem de espécies/população não muda na reação, mas os parâmetros sim
                        pop_size = len(optimizer.get('particle', [])) # Re-obtém caso algo mude
                        # A lista de espécies não é recriada aqui, será na próxima chamada a optimization
                        # Logar contagem aqui pode ser enganoso. Vamos focar nos parâmetros.
                        species_count = len(species_list_current) # Contagem da otimização anterior
                        active_count = sum(1 for s in species_list_current if s.get('Active', 1)) # Idem

                        deactivation_val = optimizer.get('CurrentDeactivation', np.nan) # Valor PÓS reação
                        shift_sev_val = optimizer.get('ShiftSeverity', np.nan) # Valor PÓS reação

                        pop_size_history_run.append(pop_size)
                        species_count_history_run.append(species_count) # Nota: contagem pré-reação
                        active_species_count_history_run.append(active_count) # Nota: contagem pré-reação
                        deactivation_history_run.append(deactivation_val)
                        shift_severity_history_run.append(shift_sev_val)
                        fe_history_run.append(current_fe_after_reaction)
                        last_logged_fe = current_fe_after_reaction
                     except Exception as log_e:
                          print(f"Aviso: Erro durante o logging pós-reação na iteração {iteration}, FE={current_fe_after_reaction}: {log_e}")


                # Imprime status após reação
                print(f"INFO: Reação concluída. Ambiente: {problem['Environmentcounter']}, FEs: {problem['FE']}/{max_evals_run}")


            # --- Critério de Parada (Redundante com While, mas seguro) ---
            if problem['FE'] >= max_evals_run:
                 print(f"\nRun {run_counter + 1} concluída. FEs totais: {problem['FE']}")
                 break

            # Checagem de segurança para loop infinito
            if iteration > max_evals_run * 2 and max_evals_run > 0: # Evita com max_evals=0
                 print(f"AVISO: Run {run_counter + 1} excedeu limite de iterações de segurança ({iteration}). Interrompendo.")
                 break
            elif iteration > 1000000 and max_evals_run <= 0: # Limite para caso sem FEs definidos
                  print(f"AVISO: Run {run_counter + 1} excedeu limite de iterações de segurança ({iteration}) sem MaxEvals definido.")
                  break


        # --- Fim da Run: Coleta de Resultados ---
        # Garante que o array de erro tenha o tamanho certo, preenchendo com NaN se parou antes
        run_current_error = problem.get('CurrentError', np.full(max_evals_total, np.nan))[:max_evals_total]
        if len(run_current_error) < max_evals_total:
            run_current_error_padded = np.full(max_evals_total, np.nan)
            run_current_error_padded[:len(run_current_error)] = run_current_error
            current_error_all_runs[run_counter, :] = run_current_error_padded
        else:
             current_error_all_runs[run_counter, :] = run_current_error

        # Calcula erro offline (média do erro corrente NAQUELA run), ignora NaNs
        offline_error_run = np.nanmean(run_current_error) if np.any(~np.isnan(run_current_error)) else np.nan
        offline_error_all_runs[run_counter] = offline_error_run

        # Armazena históricos da run (converte para array NumPy para consistência)
        all_pop_size_histories.append(np.array(pop_size_history_run))
        all_species_count_histories.append(np.array(species_count_history_run))
        all_active_species_count_histories.append(np.array(active_species_count_history_run))
        all_deactivation_histories.append(np.array(deactivation_history_run))
        all_shift_severity_histories.append(np.array(shift_severity_history_run)) # Adiciona histórico do shift
        all_fe_histories.append(np.array(fe_history_run))

        problem_final = problem # Guarda o estado final da última run executada com sucesso

        run_end_time = time.time()
        print(f"Run {run_counter + 1} levou {run_end_time - run_start_time:.2f} segundos. Erro Offline: {offline_error_run:.4e}")


    # --- Fim de Todas as Runs ---
    end_time_total = time.time()
    print(f"\nTempo total de execução para {run_number} runs: {end_time_total - start_time_total:.2f} segundos.")

    # Empacota todos os resultados em um dicionário para retorno
    run_data = {
        "problem_final": problem_final,
        "current_error": current_error_all_runs,
        "offline_error": offline_error_all_runs,
        "fe_history": all_fe_histories,
        "pop_size_history": all_pop_size_histories,
        "species_count_history": all_species_count_histories,
        "active_species_count_history": all_active_species_count_histories,
        "deactivation_history": all_deactivation_histories,
        "shift_severity_history": all_shift_severity_histories # Retorna histórico do shift
    }
    return run_data

# --- Bloco de Teste (Opcional) ---
# Este bloco só será executado se você rodar este arquivo diretamente
# Não recomendado para execução completa, use run_experiment.py
if __name__ == "__main__":
    print("\n" + "="*30)
    print("--- Teste Rápido de main_aspsogmpb ---")
    print("(Não substitui execução via run_experiment.py)")
    print("="*30)

    # Parâmetros BEM pequenos para teste rápido
    test_runs = 1
    test_peaks = 2
    test_freq = 15
    test_dim = 2
    test_shift = 0.5
    test_envs = 2
    test_max_evals = test_freq * test_envs

    print("\nExecutando com parâmetros de teste reduzidos...")
    try:
        results = main_aspsogmpb(
            peak_number=test_peaks,
            change_frequency=test_freq,
            dimension=test_dim,
            shift_severity=test_shift,
            environment_number=test_envs,
            run_number=test_runs
        )
        print("\n--- Resultados do Teste Rápido ---")
        print(f"Erro Offline Médio (Run 0): {results['offline_error'][0]:.4e}")
        print(f"Shape do Erro Corrente: {results['current_error'].shape}") # Deve ser (1, test_max_evals)
        # Verifica alguns históricos
        print(f"Número de pontos no histórico FE (Run 0): {len(results['fe_history'][0])}")
        print(f"Último FE registrado (Run 0): {results['fe_history'][0][-1] if results['fe_history'][0].size > 0 else 'N/A'}")
        print(f"Tamanho final da população (Run 0): {results['pop_size_history'][0][-1] if results['pop_size_history'][0].size > 0 else 'N/A'}")
        print(f"Número final de espécies (Run 0): {results['species_count_history'][0][-1] if results['species_count_history'][0].size > 0 else 'N/A'}")
        print(f"Último Shift Severity estimado (Run 0): {results['shift_severity_history'][0][-1] if results['shift_severity_history'][0].size > 0 else 'N/A'}")

        print("\nTeste rápido concluído com sucesso aparente.")

    except Exception as e:
        print(f"\nErro durante o teste rápido de main_aspsogmpb: {e}")
        import traceback
        traceback.print_exc()