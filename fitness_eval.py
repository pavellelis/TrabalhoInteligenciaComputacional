# fitness_eval.py
import numpy as np

# =============================================
# Função Auxiliar de Transformação GMPB
# =============================================
def _transform(X, tau, eta):
    """
    Aplica a transformação não-linear do GMPB.
    Equivalente a 'Transform' em MATLAB.

    Args:
        X (np.ndarray): Vetor de entrada (esperado como 1D ou achatado).
        tau (float): Parâmetro tau do pico.
        eta (np.ndarray): Parâmetros eta (4) do pico.

    Returns:
        np.ndarray: Vetor transformado Y (mesmo shape de X).
    """
    Y = np.zeros_like(X, dtype=float) # Garante que Y comece com zeros

    # Garante que eta tenha 4 elementos
    if not isinstance(eta, np.ndarray) or eta.size != 4:
        # Tenta converter se não for array, ou lança erro
        try:
            eta = np.array(eta).flatten()
            if eta.size != 4:
                 raise ValueError("Parâmetro eta deve ter 4 elementos")
        except Exception as e:
            raise ValueError(f"Parâmetro eta inválido ou não conversível para array de 4 elementos: {e}")


    # Processa elementos positivos
    # Adiciona pequena tolerância para evitar log(0) e problemas numéricos
    pos_mask = (X > 1e-10)
    if np.any(pos_mask):
        X_pos = X[pos_mask]
        try:
            log_X_pos = np.log(X_pos)
            # Nota: eta[0] e eta[1] são usados para X>0
            term = log_X_pos + tau * (np.sin(eta[0] * log_X_pos) + np.sin(eta[1] * log_X_pos))
            Y[pos_mask] = np.exp(term)
        except FloatingPointError: # Captura overflow/underflow/invalid em log/exp/sin
             print(f"Aviso: FloatingPointError em _transform para X > 0. X={X_pos}, tau={tau}, eta={eta[:2]}")
             Y[pos_mask] = np.inf # Ou algum outro valor indicando falha

    # Processa elementos negativos
    neg_mask = (X < -1e-10)
    if np.any(neg_mask):
        X_neg = X[neg_mask]
        try:
            # Usa log(-X_neg) que é log(|X_neg|)
            log_negX_neg = np.log(-X_neg)
            # Nota: eta[2] e eta[3] são usados para X<0
            term = log_negX_neg + tau * (np.sin(eta[2] * log_negX_neg) + np.sin(eta[3] * log_negX_neg))
            Y[neg_mask] = -np.exp(term) # Resultado é negativo
        except FloatingPointError:
             print(f"Aviso: FloatingPointError em _transform para X < 0. X={X_neg}, tau={tau}, eta={eta[2:]}")
             Y[neg_mask] = -np.inf # Ou algum outro valor indicando falha


    # Elementos próximos de zero permanecem zero em Y
    return Y

# =============================================
# Função de Fitness do GMPB
# =============================================
def fitness_aspsogmpb(X, problem):
    """
    Calcula o fitness de uma ou mais soluções X no benchmark GMPB.
    Equivalente a fitness_ASPSO_GMPB.m.

    Args:
        X (np.ndarray): Matriz de soluções (n_solutions x dimension).
                        Pode ser também um vetor 1D para uma única solução.
        problem (dict): Dicionário do estado do problema.

    Returns:
        tuple: (result, problem)
            - result (np.ndarray): Vetor coluna de fitness (n_solutions x 1).
            - problem (dict): Dicionário do problema atualizado.
    """
    # Garante que X seja sempre 2D para consistência no loop
    if X.ndim == 1:
        X = X.reshape(1, -1)
    solution_number, dimension = X.shape

    # Verifica se a dimensão da solução bate com a do problema
    if dimension != problem['Dimension']:
        raise ValueError(f"Dimensão da solução X ({dimension}) não confere com a do problema ({problem['Dimension']})")

    result = np.full((solution_number, 1), np.nan) # Inicializa resultados como NaN

    for jj in range(solution_number):
        # Verifica parada ou mudança já detectada antes de avaliar
        # Usamos >= pois FE é incrementado *após* a avaliação bem-sucedida
        if problem['FE'] >= problem['MaxEvals'] or problem.get('RecentChange', 0) == 1:
            # Não avalia mais, retorna resultados parciais e o problema como está
            # print(f"Fitness: Parando avaliação. FE={problem['FE']}, RecentChange={problem.get('RecentChange',0)}") # Debug
            return result, problem

        x_solution = X[jj, :].reshape(-1, 1) # Garante que x seja (dimension x 1)

        # Índice do ambiente atual (lembrar que Environmentcounter é 1-based)
        current_env_idx = problem['Environmentcounter'] - 1
        if not (0 <= current_env_idx < problem['EnvironmentNumber']):
             raise IndexError(f"Environmentcounter ({problem['Environmentcounter']}) fora dos limites válidos [1, {problem['EnvironmentNumber']}]")

        peak_fitnesses = np.full(problem['PeakNumber'], -np.inf) # Inicia com -inf para o max funcionar

        for k in range(problem['PeakNumber']):
            try:
                # --- Acessa dados do pico k no ambiente atual ---
                peak_pos = problem['PeaksPosition'][k, :, current_env_idx].reshape(-1, 1)
                rotation_matrix = problem['RotationMatrix'][current_env_idx][:, :, k]
                peak_tau = problem['tau'][current_env_idx, k]
                peak_eta = problem['eta'][k, :, current_env_idx] # Shape (4,)
                peak_height = problem['PeaksHeight'][current_env_idx, k]
                peak_width = problem['PeaksWidth'][k, :, current_env_idx] # Shape (dimension,)

                # --- Calcula componentes a e b ---
                diff = x_solution - peak_pos # (dimension x 1)

                # a = Transform((x - Pk)'* Rk', tau, eta) -> (1 x dim)
                a_input = (diff.T @ rotation_matrix.T).flatten()
                a = _transform(a_input, peak_tau, peak_eta) # Retorna (dimension,)

                # b = Transform(Rk * (x - Pk), tau, eta) -> (dim x 1)
                b_input = (rotation_matrix @ diff).flatten()
                b = _transform(b_input, peak_tau, peak_eta) # Retorna (dimension,)

                # --- Calcula contribuição do pico k ---
                # f(k) = H - sqrt( a * diag(Width^2) * b )
                width_sq_diag = np.diag(np.square(peak_width)) # (dim x dim)

                # (1 x dim) @ (dim x dim) @ (dim x 1) -> scalar
                term_matrix = a.reshape(1, -1) @ width_sq_diag @ b.reshape(-1, 1)
                term = term_matrix[0, 0]

                # Evita sqrt de negativo e lida com NaN/Inf que podem vir de _transform
                if np.isnan(term) or np.isinf(term) or term < 0:
                    # Se o termo é inválido, a contribuição desse pico é -infinito?
                    # Ou a altura? O código original não trata isso explicitamente.
                    # Vamos assumir que um termo inválido torna a contribuição mínima.
                    peak_fitnesses[k] = -np.inf
                    # print(f"Aviso: Termo inválido ({term}) no cálculo do pico {k}, env {current_env_idx+1}") # Debug
                else:
                    peak_fitnesses[k] = peak_height - np.sqrt(term)

            except IndexError:
                print(f"Erro de índice acessando dados para pico {k}, ambiente {current_env_idx+1}")
                peak_fitnesses[k] = -np.inf # Define como inválido
            except Exception as e:
                print(f"Erro inesperado no cálculo do pico {k}, ambiente {current_env_idx+1}: {e}")
                peak_fitnesses[k] = -np.inf # Define como inválido


        # Fitness da solução é o máximo sobre todos os picos
        # Se todos os picos deram -np.inf, o resultado será -np.inf
        final_fitness = np.max(peak_fitnesses)
        result[jj, 0] = final_fitness

        # --- Atualiza Contagem e Erro ---
        # Só incrementa FE se a avaliação foi bem-sucedida (fitness não é NaN?)
        # Vamos assumir que mesmo -inf é um resultado válido da avaliação.
        problem['FE'] += 1
        current_fe_idx = problem['FE'] - 1 # Índice 0-based para arrays

        # Verifica se excedeu o limite APÓS incrementar
        if current_fe_idx >= problem['MaxEvals']:
             print(f"Aviso: FE ({problem['FE']}) excedeu MaxEvals ({problem['MaxEvals']}) após incremento.")
             # Desfaz incremento e retorna para evitar escrita fora dos limites
             problem['FE'] -= 1
             # O 'result' calculado para jj ainda é válido, mas não avançamos mais
             return result, problem

        # Erro da solução atual
        # Verifica se OptimumValue é válido antes de subtrair
        optimum_value_current_env = problem['OptimumValue'][current_env_idx]
        if np.isnan(optimum_value_current_env) or np.isnan(final_fitness):
             solution_error = np.nan # Erro indefinido se ótimo ou fitness for NaN
        else:
             solution_error = optimum_value_current_env - final_fitness

        # Atualiza CurrentError (mantendo o melhor erro dentro do ambiente)
        # Lógica do MATLAB: if rem(FE,Freq)~=1 ... else ...
        # Equivalente a: if FE % Freq != 1 (ou seja, não é o primeiro FE do ambiente)
        # Mas FE começa em 1 no MATLAB. Em Python, FE=1 é o primeiro.
        # A lógica é: se for o primeiro FE *neste ambiente*, apenas grave o erro.
        # Caso contrário, compare com o erro anterior e mantenha o menor.
        is_first_fe_in_env = (problem['FE'] == 1) or (problem['FE'] % problem['ChangeFrequency'] == 1)

        if is_first_fe_in_env:
            problem['CurrentError'][current_fe_idx] = solution_error
        else:
            # Garante que o índice anterior seja válido
            if current_fe_idx > 0:
                prev_error = problem['CurrentError'][current_fe_idx - 1]
                # Se o erro anterior for NaN, usamos o erro atual
                if np.isnan(prev_error):
                     problem['CurrentError'][current_fe_idx] = solution_error
                # Se o erro atual for NaN, mantemos o anterior (que não é NaN)
                elif np.isnan(solution_error):
                     problem['CurrentError'][current_fe_idx] = prev_error
                # Ambos são válidos, pegamos o mínimo
                else:
                     problem['CurrentError'][current_fe_idx] = min(prev_error, solution_error)
            else: # Caso de current_fe_idx == 0 mas não é first_fe_in_env (não deve acontecer)
                problem['CurrentError'][current_fe_idx] = solution_error


        # --- Verifica Mudança de Ambiente ---
        # Checa se o FE atual é múltiplo da frequência E se ainda não atingiu o máximo
        if problem['FE'] % problem['ChangeFrequency'] == 0 and problem['FE'] < problem['MaxEvals']:
            problem['Environmentcounter'] += 1
            problem['RecentChange'] = 1
            # Não retorna aqui; a flag fará o loop externo em main_aspsogmpb reagir

    return result, problem