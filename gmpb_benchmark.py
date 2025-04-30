import numpy as np


def _rotation(theta, dimension):

    if dimension == 1:
        return np.identity(1)

    page_number = dimension * (dimension - 1) // 2
    # Checa se page_number é zero (acontece se dimension < 2)
    if page_number <= 0:
        return np.identity(dimension)

    elementary_rotations = []  # Lista para guardar matrizes elementares

    for ii in range(dimension):
        for jj in range(ii + 1, dimension):
            tmp_matrix = np.identity(dimension)
            tmp_matrix[ii, ii] = np.cos(theta)
            tmp_matrix[jj, jj] = np.cos(theta)
            tmp_matrix[ii, jj] = -np.sin(theta)  # Convenção comum de rotação
            tmp_matrix[jj, ii] = np.sin(theta)  # Convenção comum de rotação
            elementary_rotations.append(tmp_matrix)

    output_matrix = np.identity(dimension)
    # Permuta a ordem das rotações elementares e as aplica
    # np.random.permutation(page_number) gera índices de 0 a page_number-1
    permuted_indices = np.random.permutation(page_number)

    for idx in permuted_indices:
        # Garante que idx esteja dentro dos limites da lista
        if idx < len(elementary_rotations):
            output_matrix = output_matrix @ elementary_rotations[idx]  # Multiplicação de matrizes

    return output_matrix


# =============================================
# Gerador do Benchmark GMPB
# =============================================
def benchmark_generator_aspsogmpb(peak_number, change_frequency, dimension,
                                  shift_severity, environment_number):
    """
    Gera os parâmetros e estados para o benchmark GMPB.
    Equivalente a BenchmarkGenerator_ASPSO_GMPB.m.

    Args:
        peak_number (int): Número de picos.
        change_frequency (int): Avaliações de função por ambiente.
        dimension (int): Dimensionalidade do espaço.
        shift_severity (float): Magnitude da mudança de posição dos picos.
        environment_number (int): Número total de ambientes.

    Returns:
        dict: Dicionário contendo o estado e parâmetros do problema ('Problem').
    """
    problem = {}
    problem['FE'] = 0
    problem['PeakNumber'] = peak_number
    problem['ChangeFrequency'] = change_frequency
    problem['Dimension'] = dimension
    problem['ShiftSeverity'] = shift_severity
    problem['EnvironmentNumber'] = environment_number
    problem['Environmentcounter'] = 1  # Contador começa em 1 como no MATLAB
    problem['RecentChange'] = 0  # Flag: 0 = não, 1 = sim
    problem['MaxEvals'] = problem['ChangeFrequency'] * problem['EnvironmentNumber']

    # Inicializa arrays para guardar histórico/estados
    problem['Ebbc'] = np.full(problem['EnvironmentNumber'], np.nan)
    problem['CurrentError'] = np.full(problem['MaxEvals'], np.nan)
    problem['CurrentPerformance'] = np.full(problem['MaxEvals'], np.nan)

    # Limites e parâmetros dos picos
    problem['MinCoordinate'] = -100.0
    problem['MaxCoordinate'] = 100.0
    problem['MinHeight'] = 30.0
    problem['MaxHeight'] = 70.0
    problem['MinWidth'] = 1.0
    problem['MaxWidth'] = 12.0
    problem['MinAngle'] = -np.pi
    problem['MaxAngle'] = np.pi
    problem['MinTau'] = -1.0
    problem['MaxTau'] = 1.0
    problem['MinEta'] = -20.0
    problem['MaxEta'] = 20.0

    # Severidades das mudanças
    problem['HeightSeverity'] = 7.0
    problem['WidthSeverity'] = 1.0
    problem['AngleSeverity'] = np.pi / 9.0
    problem['TauSeverity'] = 0.2
    problem['EtaSeverity'] = 2.0

    # Inicializa arrays para guardar características dos picos por ambiente
    problem['OptimumValue'] = np.full(problem['EnvironmentNumber'], np.nan)
    problem['OptimumID'] = np.full(problem['EnvironmentNumber'], np.nan)  # Índice do pico ótimo
    problem['PeaksHeight'] = np.full((problem['EnvironmentNumber'], problem['PeakNumber']), np.nan)
    problem['PeaksPosition'] = np.full((problem['PeakNumber'], problem['Dimension'], problem['EnvironmentNumber']),
                                       np.nan)
    problem['PeaksWidth'] = np.full((problem['PeakNumber'], problem['Dimension'], problem['EnvironmentNumber']), np.nan)
    problem['PeaksAngle'] = np.full((problem['EnvironmentNumber'], problem['PeakNumber']), np.nan)
    problem['tau'] = np.full((problem['EnvironmentNumber'], problem['PeakNumber']), np.nan)
    problem['eta'] = np.full((problem['PeakNumber'], 4, problem['EnvironmentNumber']), np.nan)

    # --- Inicialização do Ambiente 1 ---
    env_idx = 0  # Índice 0 para o primeiro ambiente em Python

    # Posições Iniciais
    problem['PeaksPosition'][:, :, env_idx] = problem['MinCoordinate'] + \
                                              (problem['MaxCoordinate'] - problem['MinCoordinate']) * \
                                              np.random.rand(problem['PeakNumber'], problem['Dimension'])

    # Alturas Iniciais
    problem['PeaksHeight'][env_idx, :] = problem['MinHeight'] + \
                                         (problem['MaxHeight'] - problem['MinHeight']) * \
                                         np.random.rand(problem['PeakNumber'])

    # Larguras Iniciais
    problem['PeaksWidth'][:, :, env_idx] = problem['MinWidth'] + \
                                           (problem['MaxWidth'] - problem['MinWidth']) * \
                                           np.random.rand(problem['PeakNumber'], problem['Dimension'])

    # Ângulos Iniciais
    problem['PeaksAngle'][env_idx, :] = problem['MinAngle'] + \
                                        (problem['MaxAngle'] - problem['MinAngle']) * \
                                        np.random.rand(problem['PeakNumber'])

    # Tau Iniciais
    problem['tau'][env_idx, :] = problem['MinTau'] + \
                                 (problem['MaxTau'] - problem['MinTau']) * \
                                 np.random.rand(problem['PeakNumber'])

    # Eta Iniciais
    problem['eta'][:, :, env_idx] = problem['MinEta'] + \
                                    (problem['MaxEta'] - problem['MinEta']) * \
                                    np.random.rand(problem['PeakNumber'], 4)

    # Ótimo Inicial
    problem['OptimumValue'][env_idx] = np.max(problem['PeaksHeight'][env_idx, :])
    problem['OptimumID'][env_idx] = np.argmax(problem['PeaksHeight'][env_idx, :])

    # Matrizes de Rotação Iniciais
    problem['EllipticalPeaks'] = 1
    problem['InitialRotationMatrix'] = np.full((problem['Dimension'], problem['Dimension'], problem['PeakNumber']),
                                               np.nan)
    for ii in range(problem['PeakNumber']):
        # --- ALTERAÇÃO AQUI ---
        # Lógica Original (usando scipy.linalg.qr):
        # Q, _ = scipy_qr(np.random.rand(problem['Dimension'], problem['Dimension']))
        # problem['InitialRotationMatrix'][:, :, ii] = Q

        # Lógica Alternativa (usando numpy.linalg.svd):
        # 1. Cria uma matriz aleatória quadrada.
        random_matrix = np.random.rand(problem['Dimension'], problem['Dimension'])
        # 2. Aplica a Decomposição em Valores Singulares (SVD).
        #    A = U * Sigma * Vh (onde U e Vh são ortogonais/unitárias)
        try:
            U, _, _ = np.linalg.svd(random_matrix)
            # 3. Usa a matriz ortogonal U como a matriz de rotação/reflexão inicial.
            #    (Vh.T também seria uma opção ortogonal válida)
            problem['InitialRotationMatrix'][:, :, ii] = U
        except np.linalg.LinAlgError:
            # Em casos raros, SVD pode falhar se a matriz for singular/mal condicionada
            # Como fallback, podemos usar uma matriz identidade ou tentar novamente.
            print(f"Aviso: SVD falhou para o pico {ii}. Usando matriz identidade como fallback.")
            problem['InitialRotationMatrix'][:, :, ii] = np.identity(problem['Dimension'])
        # --- FIM DA ALTERAÇÃO ---

    # Armazena rotações por ambiente
    problem['RotationMatrix'] = [None] * problem['EnvironmentNumber']
    problem['RotationMatrix'][env_idx] = problem['InitialRotationMatrix'].copy()

    # --- Geração dos Ambientes Subsequentes (2 a EnvironmentNumber) ---
    for ii in range(1, problem['EnvironmentNumber']):
        prev_env_idx = ii - 1
        curr_env_idx = ii

        # Calcula vetor de deslocamento aleatório normalizado
        shift_offset = np.random.randn(problem['PeakNumber'], problem['Dimension'])
        norms = np.linalg.norm(shift_offset, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Evita divisão por zero
        shift = (shift_offset / norms) * problem['ShiftSeverity']

        # Atualiza Posições, Larguras, Alturas, Ângulos, Tau, Eta
        peaks_position = problem['PeaksPosition'][:, :, prev_env_idx] + shift
        peaks_width = problem['PeaksWidth'][:, :, prev_env_idx] + \
                      (np.random.randn(problem['PeakNumber'], problem['Dimension']) * problem['WidthSeverity'])
        peaks_height = problem['PeaksHeight'][prev_env_idx, :] + \
                       (problem['HeightSeverity'] * np.random.randn(problem['PeakNumber']))
        peaks_angle = problem['PeaksAngle'][prev_env_idx, :] + \
                      (problem['AngleSeverity'] * np.random.randn(problem['PeakNumber']))
        peaks_tau = problem['tau'][prev_env_idx, :] + \
                    (problem['TauSeverity'] * np.random.randn(problem['PeakNumber']))
        peaks_eta = problem['eta'][:, :, prev_env_idx] + \
                    (np.random.randn(problem['PeakNumber'], 4) * problem['EtaSeverity'])

        # --- Aplica Limites (Reflexão) ---
        # (Lógica de aplicação de limites permanece a mesma)
        # Ângulos
        mask = peaks_angle > problem['MaxAngle']
        peaks_angle[mask] = (2 * problem['MaxAngle']) - peaks_angle[mask]
        mask = peaks_angle < problem['MinAngle']
        peaks_angle[mask] = (2 * problem['MinAngle']) - peaks_angle[mask]
        # Tau
        mask = peaks_tau > problem['MaxTau']
        peaks_tau[mask] = (2 * problem['MaxTau']) - peaks_tau[mask]
        mask = peaks_tau < problem['MinTau']
        peaks_tau[mask] = (2 * problem['MinTau']) - peaks_tau[mask]
        # Eta
        mask = peaks_eta > problem['MaxEta']
        peaks_eta[mask] = (2 * problem['MaxEta']) - peaks_eta[mask]
        mask = peaks_eta < problem['MinEta']
        peaks_eta[mask] = (2 * problem['MinEta']) - peaks_eta[mask]
        # Posições
        mask = peaks_position > problem['MaxCoordinate']
        peaks_position[mask] = (2 * problem['MaxCoordinate']) - peaks_position[mask]
        mask = peaks_position < problem['MinCoordinate']
        peaks_position[mask] = (2 * problem['MinCoordinate']) - peaks_position[mask]
        # Alturas
        mask = peaks_height > problem['MaxHeight']
        peaks_height[mask] = (2 * problem['MaxHeight']) - peaks_height[mask]
        mask = peaks_height < problem['MinHeight']
        peaks_height[mask] = (2 * problem['MinHeight']) - peaks_height[mask]
        # Larguras
        mask = peaks_width > problem['MaxWidth']
        peaks_width[mask] = (2 * problem['MaxWidth']) - peaks_width[mask]
        mask = peaks_width < problem['MinWidth']
        peaks_width[mask] = (2 * problem['MinWidth']) - peaks_width[mask]

        # Armazena novos valores no problema
        problem['PeaksPosition'][:, :, curr_env_idx] = peaks_position
        problem['PeaksHeight'][curr_env_idx, :] = peaks_height
        problem['PeaksWidth'][:, :, curr_env_idx] = peaks_width
        problem['PeaksAngle'][curr_env_idx, :] = peaks_angle
        problem['tau'][curr_env_idx, :] = peaks_tau
        problem['eta'][:, :, curr_env_idx] = peaks_eta

        # Atualiza Matrizes de Rotação
        current_rotations = np.zeros_like(problem['InitialRotationMatrix'])
        for jj in range(problem['PeakNumber']):
            additional_rotation = _rotation(peaks_angle[jj], problem['Dimension'])
            # Multiplica a rotação inicial pela rotação adicional
            current_rotations[:, :, jj] = problem['InitialRotationMatrix'][:, :, jj] @ additional_rotation

        problem['RotationMatrix'][curr_env_idx] = current_rotations

        # Encontra Ótimo do novo ambiente
        problem['OptimumValue'][curr_env_idx] = np.max(peaks_height)
        problem['OptimumID'][curr_env_idx] = np.argmax(peaks_height)  # Índice 0-based

    # Reseta o contador de ambiente para ser usado pelo otimizador
    problem['Environmentcounter'] = 1  # Começa em 1 para corresponder ao acesso no fitness

    return problem


# --- Bloco de Teste ---
# (Este bloco só será executado se você rodar este arquivo diretamente)
if __name__ == "__main__":
    print("--- Testando a função _rotation ---")
    # Testa _rotation com theta em radianos (ex: pi/6 = 30 graus)
    theta_rad = np.pi / 6
    dimension = 3
    try:
        rotation_matrix = _rotation(theta_rad, dimension)
        print(f"Matriz de Rotação para theta={theta_rad:.4f} rad, D={dimension}:\n{rotation_matrix}")
        # Verifica se é aproximadamente ortogonal: R @ R.T ≈ I
        identity_check = rotation_matrix @ rotation_matrix.T
        print(f"\nVerificação de Ortogonalidade (R @ R.T deve ser aprox. I):\n{identity_check}")
        if np.allclose(identity_check, np.identity(dimension)):
            print("-> Matriz de rotação parece ortogonal.")
        else:
            print("-> AVISO: Matriz de rotação pode não ser perfeitamente ortogonal.")

    except Exception as e:
        print(f"Erro ao testar _rotation: {e}")

    print("\n--- Testando a função benchmark_generator_aspsogmpb ---")
    # Parâmetros de teste
    test_peak_number = 2
    test_change_frequency = 10
    test_dimension = 2
    test_shift_severity = 1.0
    test_environment_number = 3
    try:
        np.random.seed(42)  # Para reprodutibilidade do teste
        test_problem = benchmark_generator_aspsogmpb(
            test_peak_number, test_change_frequency, test_dimension,
            test_shift_severity, test_environment_number
        )
        print(f"Benchmark gerado com sucesso.")
        print(f"Chaves no dicionário 'problem': {list(test_problem.keys())}")
        print(f"Shape de PeaksPosition: {test_problem['PeaksPosition'].shape}")  # Deve ser (peak, dim, env)
        print(f"Shape de PeaksHeight: {test_problem['PeaksHeight'].shape}")  # Deve ser (env, peak)
        print(f"Shape de RotationMatrix[0]: {test_problem['RotationMatrix'][0].shape}")  # Deve ser (dim, dim, peak)
        print(f"OptimumValue (primeiros {test_environment_number} ambientes): {test_problem['OptimumValue']}")
        print(f"OptimumID (primeiros {test_environment_number} ambientes): {test_problem['OptimumID']}")
        print(f"Posição do Pico 0, Ambiente 0: {test_problem['PeaksPosition'][0, :, 0]}")

    except Exception as e:
        print(f"Erro ao testar benchmark_generator_aspsogmpb: {e}")
        import traceback

        traceback.print_exc()