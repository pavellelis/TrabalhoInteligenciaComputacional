# aspsogmpb_core.py
import numpy as np

# Importa funções auxiliares e de fitness dos módulos anteriores
try:
    from aspsogmpb_utils import creating_species, initializing_optimizer_aspsogmpb
    from fitness_eval import fitness_aspsogmpb
except ImportError as e:
    print(f"Erro ao importar módulos em aspsogmpb_core.py: {e}")
    # Define placeholders para permitir a análise sintática, mas falhará na execução
    def creating_species(opt): return [], opt
    def initializing_optimizer_aspsogmpb(lb, ub, dim, prob): return {'X':np.array([[0]]), 'PbestPosition':np.array([[0]]), 'Velocity':np.array([[0]]), 'FitnessValue':0, 'PbestFitness':0}, prob
    def fitness_aspsogmpb(X, prob): return np.zeros((X.shape[0],1)), prob
    # Define reaction_aspsogmpb aqui também se for importá-la internamente (não é o caso)

# =============================================
# Iteração Principal do ASPSO (Optimization Step)
# =============================================
def optimization_aspsogmpb(optimizer, problem):
    """
    Executa uma iteração do algoritmo ASPSO+AP+AD (passo de otimização).
    Equivalente à lógica principal dentro do loop em Optimization_ASPSO_GMPB.m.

    Args:
        optimizer (dict): Estado atual do otimizador.
        problem (dict): Estado atual do benchmark.

    Returns:
        tuple: (optimizer, problem, species_list)
            - optimizer (dict): Estado atualizado do otimizador.
            - problem (dict): Estado atualizado do benchmark.
            - species_list (list): Espécies identificadas/atualizadas nesta iteração.
    """
    try:
        dimension = optimizer['Dimension']
        min_coord = optimizer['MinCoordinate']
        max_coord = optimizer['MaxCoordinate']
    except KeyError as e:
        raise KeyError(f"Parâmetro essencial faltando no dicionário 'optimizer': {e}")

    # --- 1. Criação de Espécies e Identificação de Trackers ---
    num_pre_iteration_tracker = len(optimizer.get('tracker', [])) # Trackers da iteração anterior
    species_list, optimizer = creating_species(optimizer)
    current_tracker_indices = [] # Lista de ÍNDICES (na species_list ATUAL) das espécies tracker
    best_tracker_fitness = -np.inf
    best_tracker_idx_in_list = -1 # Índice do melhor tracker na species_list atual

    tracker_threshold = optimizer.get('teta', 1.0)
    particles = optimizer.get('particle', []) # Atalho

    for i, species in enumerate(species_list):
        # Checa se é tracker (convergida)
        species_distance = species.get('distance', np.inf) # Assume não convergida se distância ausente
        if not np.isnan(species_distance) and species_distance < tracker_threshold:
            current_tracker_indices.append(i)
            # Encontra o melhor tracker (maior PbestFitness da semente)
            try:
                seed_fitness = particles[species['seed']]['PbestFitness']
                if not np.isnan(seed_fitness) and seed_fitness > best_tracker_fitness:
                   best_tracker_fitness = seed_fitness
                   best_tracker_idx_in_list = i
            except (KeyError, IndexError):
                 print(f"Aviso (Opt): Erro ao acessar PbestFitness da semente {species.get('seed', 'N/A')} da espécie {i}")
                 continue # Pula esta espécie se dados da semente estiverem ruins

    optimizer['tracker'] = current_tracker_indices # Atualiza a lista de trackers no otimizador

    # --- 2. Exclusão ---
    removed_species_indices_in_list = set() # Índices (na species_list) a remover
    removed_particle_indices = set()        # Índices GLOBAIS das partículas a remover

    num_species = len(species_list)
    exclusion_limit = optimizer.get('ExclusionLimit', 0.5)

    for ii in range(num_species):
        if ii in removed_species_indices_in_list: continue
        try:
            seed_ii_idx = species_list[ii]['seed']
            seed_ii_pbest_pos = particles[seed_ii_idx]['PbestPosition'].flatten()
            seed_ii_pbest_fit = particles[seed_ii_idx]['PbestFitness']
        except (KeyError, IndexError): continue

        for jj in range(ii + 1, num_species):
            if jj in removed_species_indices_in_list or ii in removed_species_indices_in_list: continue
            try:
                seed_jj_idx = species_list[jj]['seed']
                seed_jj_pbest_pos = particles[seed_jj_idx]['PbestPosition'].flatten()
                seed_jj_pbest_fit = particles[seed_jj_idx]['PbestFitness']
            except (KeyError, IndexError): continue

            # Calcula distância entre sementes
            distance = np.linalg.norm(seed_ii_pbest_pos - seed_jj_pbest_pos)

            if distance < exclusion_limit:
                # Marca a pior para remoção
                if seed_ii_pbest_fit < seed_jj_pbest_fit:
                    removed_species_indices_in_list.add(ii)
                    removed_particle_indices.update(species_list[ii]['member'])
                    break # Espécie ii removida, passa para próxima ii
                else:
                    removed_species_indices_in_list.add(jj)
                    removed_particle_indices.update(species_list[jj]['member'])
                    # Não precisa de break aqui, continua comparando ii com outras jj

    # --- 3. Atualiza Trackers após Exclusão ---
    # Filtra a lista de trackers mantendo apenas os índices que NÃO foram marcados para remoção
    optimizer['tracker'] = [idx for idx in optimizer['tracker'] if idx not in removed_species_indices_in_list]
    num_current_iteration_tracker = len(optimizer['tracker'])

    # --- 4. Lógica de Desativação Adaptativa (AD) ---
    # Se número de trackers aumentou (relativo à iteração anterior), reseta Raio de Desativação
    # Nota: O artigo original pode ter lógica ligeiramente diferente aqui (comparar < ou >?). Vamos seguir a do código MATLAB.
    if num_current_iteration_tracker > num_pre_iteration_tracker:
        optimizer['CurrentDeactivation'] = optimizer.get('MaxDeactivation', 0.7) # Reset
        optimizer['beta'] = 1.0 # Reseta beta também? O código MATLAB parece não fazer. Confirmar. Não reseta beta aqui.

    # Verifica convergência dos trackers atuais e atualiza suas flags 'Active'
    all_trackers_converged_check = True if num_current_iteration_tracker > 0 else False
    deactivation_threshold = optimizer.get('CurrentDeactivation', 0.7)
    active_tracker_indices_now = [] # Trackers que permanecem ativos *nesta iteração*

    if optimizer['tracker']: # Se houver trackers
        # Atualiza a flag 'Active' na species_list original (antes de remover espécies)
        for tracker_species_idx in optimizer['tracker']:
            # Verifica se o índice ainda é válido após possível remoção de espécies anterior
            if tracker_species_idx < len(species_list):
                species = species_list[tracker_species_idx]
                species_distance = species.get('distance', np.inf)

                if np.isnan(species_distance) or species_distance > deactivation_threshold:
                    all_trackers_converged_check = False
                    species['Active'] = 1 # Garante que está ativa
                    active_tracker_indices_now.append(tracker_species_idx)
                else:
                    # Convergiu (<= deactivation_threshold)
                    # Desativa SE NÃO for o melhor tracker global
                    if tracker_species_idx != best_tracker_idx_in_list:
                        species['Active'] = 0 # Marca como inativo
                    else:
                        species['Active'] = 1 # Melhor tracker sempre ativo
                        active_tracker_indices_now.append(tracker_species_idx) # Melhor ainda está ativo
            else:
                 # Índice inválido, tracker provavelmente foi removido por exclusão
                 all_trackers_converged_check = False # Assume não convergido se tracker desapareceu


    # Se todos os trackers convergiram (e existia algum), atualiza (diminui) o limiar r_a
    if all_trackers_converged_check and num_current_iteration_tracker > 0:
        min_deactivation = optimizer.get('MinDeactivation', 0.1)
        if optimizer['CurrentDeactivation'] > min_deactivation:
             optimizer['beta'] = optimizer.get('beta', 1.0) * optimizer.get('gama', 0.1)
             optimizer['CurrentDeactivation'] = min_deactivation + \
                 (optimizer.get('MaxDeactivation', 0.7) - min_deactivation) * optimizer['beta']
             # Reativa todos os trackers (exceto talvez o melhor?) para forçar mais convergência com r_a menor?
             # Algorithm 2 (linha 30) diz "Activate all species". Vamos seguir.
             # Isso pode sobrescrever a desativação feita acima se não for o melhor.
             # No entanto, a lógica de desativação (linhas 31-35) vem DEPOIS.
             # Vamos seguir a ordem do Algorithm 2:
             # 1. Resetar r_a se novo tracker (feito acima, implicitamente ao não constringir)
             # 2. CONSTRINGIR r_a se todos convergiram (feito acima)
             # 3. ATIVAR todos (?) - Linha 30 Algo2. Parece contraditório com desativação a seguir.
             # 4. DESATIVAR se s <= r_a (exceto melhor) - Linhas 31-35 Algo2
             # Vamos omitir o "Activate all species" (Linha 30) pois parece conflitar com a desativação seletiva logo após.
             # A lógica já implementada acima (marcar Active=0/1) parece mais consistente com a Seção 3.2.


    # --- 5. Remove Espécies Marcadas por Exclusão ---
    # Cria a nova lista de espécies, mantendo apenas as que não foram marcadas para remoção
    species_list_after_removal = [species for i, species in enumerate(species_list) if i not in removed_species_indices_in_list]
    species_list = species_list_after_removal # Atualiza a lista principal
    num_species_after_removal = len(species_list)

    # --- 6. Atualiza Raios de Exclusão e Geração ---
    # Atualiza baseado no número ATUAL de espécies
    if num_species_after_removal > 0 and dimension > 0:
        # d_boa = (UB - LB) / (N ^ (1/D))
        range_coord = optimizer['MaxCoordinate'] - optimizer['MinCoordinate']
        n_pow_inv_d = num_species_after_removal ** (1.0 / dimension)
        d_boa = range_coord / max(n_pow_inv_d, 1e-9) # Evita divisão por zero
        optimizer['ExclusionLimit'] = 0.5 * d_boa
        optimizer['GenerateRadious'] = 0.3 * d_boa
    else:
        # Caso sem espécies ou dim=0, usa valor grande ou default? Usa default grande.
        range_coord = optimizer['MaxCoordinate'] - optimizer['MinCoordinate']
        optimizer['ExclusionLimit'] = 0.5 * range_coord
        optimizer['GenerateRadious'] = 0.3 * range_coord


    # --- 7. Anti-Convergência / Reinicialização ---
    # Verifica se TODAS as espécies restantes estão convergidas (usando GenerateRadius)
    all_converged_check = True
    if num_species_after_removal == 0:
         all_converged_check = False
    else:
        generate_radius = optimizer.get('GenerateRadious', 0.3)
        for species in species_list:
             species_distance = species.get('distance', np.inf)
             if np.isnan(species_distance) or species_distance > generate_radius:
                  all_converged_check = False
                  break

    # Ativa anti-convergência se N >= Nmax E all_converged_check é True
    n_max = optimizer.get('Nmax', 30)
    if num_species_after_removal >= n_max and all_converged_check:
        print(f"INFO (Opt): Anti-convergência ativado (N={num_species_after_removal} >= Nmax={n_max} e todos convergiram)")
        worst_swarm_fitness = np.inf
        worst_swarm_idx_in_list = -1
        # Encontra a pior espécie (menor PbestFitness da semente)
        for i, species in enumerate(species_list):
             try:
                 seed_fit = particles[species['seed']]['PbestFitness']
                 if not np.isnan(seed_fit) and seed_fit < worst_swarm_fitness:
                      worst_swarm_fitness = seed_fit
                      worst_swarm_idx_in_list = i
             except (KeyError, IndexError): continue

        if worst_swarm_idx_in_list != -1:
            worst_species_members_indices = species_list[worst_swarm_idx_in_list]['member']
            print(f"INFO (Opt): Reinicializando membros {worst_species_members_indices} da pior espécie {worst_swarm_idx_in_list}")

            # Remove as partículas antigas da pior espécie
            indices_to_reinit_set = set(worst_species_members_indices)
            particles_kept = [p for i, p in enumerate(particles) if i not in indices_to_reinit_set]

            # Cria novas partículas para substituir
            new_particles_for_worst = []
            problem_after_reinit = problem # Passa o estado atual do problema
            num_to_reinit = len(worst_species_members_indices)

            for _ in range(num_to_reinit):
                 new_particle, problem_after_reinit = initializing_optimizer_aspsogmpb(
                     min_coord, max_coord, dimension, problem_after_reinit)

                 # Se a inicialização causou mudança, precisa tratar imediatamente
                 if problem_after_reinit.get('RecentChange', 0):
                      print("AVISO (Opt): Mudança detectada durante anti-convergência!")
                      # Adiciona as novas partículas criadas até agora
                      optimizer['particle'] = particles_kept + new_particles_for_worst + [new_particle]
                      # Remove partículas marcadas por exclusão ANTES de recriar espécies
                      if removed_particle_indices:
                           optimizer['particle'] = [p for i, p in enumerate(optimizer['particle']) if i not in removed_particle_indices]
                      # Recria espécies e retorna
                      species_list, optimizer = creating_species(optimizer)
                      return optimizer, problem_after_reinit, species_list

                 new_particles_for_worst.append(new_particle)

            # Atualiza a lista de partículas do otimizador
            optimizer['particle'] = particles_kept + new_particles_for_worst
            particles = optimizer['particle'] # Atualiza o atalho

            # Recria as espécies pois a população mudou
            print("INFO (Opt): Recriando espécies após anti-convergência.")
            species_list, optimizer = creating_species(optimizer)
            # A anti-convergência pode ter mudado quais espécies são trackers, etc.
            # Reavaliar trackers? Não, deixa para a próxima iteração.
            num_species_after_removal = len(species_list) # Atualiza contagem

    # --- 8. Executa Passo do PSO ---
    print(f"INFO (Opt): Executando PSO para {sum(s.get('Active',0) for s in species_list)} de {num_species_after_removal} espécies.")
    constriction_factor = optimizer.get('x', 0.729)
    c1 = optimizer.get('c1', 2.05)
    c2 = optimizer.get('c2', 2.05)

    for i, species in enumerate(species_list):
        if species.get('Active', 1): # Processa apenas espécies ativas
             try:
                 seed_idx = species['seed']
                 seed_pbest_pos = particles[seed_idx]['PbestPosition']
                 # Garante que seed_pbest_pos tem shape (1, dimension) para broadcasting
                 if seed_pbest_pos.ndim == 1: seed_pbest_pos = seed_pbest_pos.reshape(1, -1)
             except (KeyError, IndexError):
                  print(f"Aviso (Opt): Não foi possível obter Pbest da semente {species.get('seed', 'N/A')} para espécie {i}. Pulando PSO.")
                  continue

             for member_idx in species['member']:
                 try:
                     particle = particles[member_idx]

                     # Garante que X, Velocity, PbestPosition tenham shape (1, dimension)
                     if particle['X'].ndim == 1: particle['X'] = particle['X'].reshape(1, -1)
                     if particle['Velocity'].ndim == 1: particle['Velocity'] = particle['Velocity'].reshape(1, -1)
                     if particle['PbestPosition'].ndim == 1: particle['PbestPosition'] = particle['PbestPosition'].reshape(1, -1)


                     # Atualiza Velocidade
                     rand1 = np.random.rand(1, dimension)
                     rand2 = np.random.rand(1, dimension)
                     cognitive_comp = c1 * rand1 * (particle['PbestPosition'] - particle['X'])
                     social_comp = c2 * rand2 * (seed_pbest_pos - particle['X']) # Usa Pbest da semente
                     new_velocity = constriction_factor * (particle['Velocity'] + cognitive_comp + social_comp)

                     # Atualiza Posição
                     new_position = particle['X'] + new_velocity

                     # --- Tratamento de Limites ---
                     clamped_position = np.clip(new_position, min_coord, max_coord)
                     # Identifica onde houve clamp para zerar velocidade
                     clamped_mask = (new_position < min_coord) | (new_position > max_coord)
                     new_velocity[clamped_mask] = 0.0

                     # Atualiza estado da partícula (posição e velocidade)
                     particle['X'] = clamped_position
                     particle['Velocity'] = new_velocity

                     # --- Avalia Fitness da Nova Posição ---
                     fitness_arr, problem = fitness_aspsogmpb(particle['X'], problem)
                     # Se fitness_arr for vazio ou NaN, define como -inf
                     current_fitness = fitness_arr[0, 0] if fitness_arr.size > 0 and not np.isnan(fitness_arr[0,0]) else -np.inf
                     particle['FitnessValue'] = current_fitness

                     # --- Verifica Mudança Durante Fitness ---
                     if problem.get('RecentChange', 0):
                          print("AVISO (Opt): Mudança detectada durante avaliação de fitness no PSO!")
                          # Remove partículas marcadas por exclusão ANTES de recriar espécies
                          if removed_particle_indices:
                               print(f"INFO (Opt): Removendo {len(removed_particle_indices)} partículas excluídas antes de retornar.")
                               optimizer['particle'] = [p for i, p in enumerate(particles) if i not in removed_particle_indices]
                               particles = optimizer['particle'] # Atualiza atalho
                          # Recria espécies e retorna imediatamente
                          species_list, optimizer = creating_species(optimizer)
                          return optimizer, problem, species_list

                     # --- Atualiza Pbest ---
                     # Compara com PbestFitness, tratando possível -inf
                     current_pbest_fitness = particle.get('PbestFitness', -np.inf)
                     if np.isnan(current_pbest_fitness): current_pbest_fitness = -np.inf

                     if current_fitness > current_pbest_fitness:
                           particle['PbestFitness'] = current_fitness
                           particle['PbestPosition'] = particle['X'].copy()

                 except (KeyError, IndexError) as e:
                     print(f"Aviso (Opt): Erro processando partícula {member_idx} na espécie {i} durante PSO: {e}")
                     # traceback.print_exc() # Descomentar para mais detalhes
                     continue # Pula para próxima partícula

    # --- 9. Remove Partículas Marcadas por Exclusão (após o PSO) ---
    # É importante fazer isso APÓS o loop PSO ter terminado sem retornos por mudança
    if removed_particle_indices:
        num_before_removal = len(particles)
        optimizer['particle'] = [p for i, p in enumerate(particles) if i not in removed_particle_indices]
        particles = optimizer['particle'] # Atualiza atalho
        num_after_removal = len(particles)
        if num_after_removal < num_before_removal:
             print(f"INFO (Opt): Removidas {num_before_removal - num_after_removal} partículas por exclusão após PSO.")
             # Recria espécies pois a população e índices mudaram
             species_list, optimizer = creating_species(optimizer)
             num_species_after_removal = len(species_list) # Atualiza contagem
             print(f"INFO (Opt): Espécies recriadas após remoção. Novas espécies: {num_species_after_removal}")
        else:
             print("Aviso (Opt): Nenhuma partícula removida por exclusão no final, embora houvesse índices marcados.")


    # --- 10. Adiciona Novas Partículas (Diversidade se Convergido) ---
    # Verifica convergência novamente com a lista de espécies atual (pós-remoção/recriação)
    all_converged_check_after_pso = True
    if num_species_after_removal == 0:
         all_converged_check_after_pso = False
    else:
        generate_radius = optimizer.get('GenerateRadious', 0.3)
        for species in species_list:
             species_distance = species.get('distance', np.inf)
             if np.isnan(species_distance) or species_distance > generate_radius:
                  all_converged_check_after_pso = False
                  break

    # Adiciona se N < Nmax E all_converged é True
    if num_species_after_removal < n_max and all_converged_check_after_pso:
        num_to_add = optimizer.get('NewlyAddedPopulationSize', 5)
        print(f"INFO (Opt): Adicionando {num_to_add} novas partículas (N={num_species_after_removal} < Nmax={n_max} e todos convergiram).")
        problem_after_add = problem # Passa estado atual do problema
        added_particles_temp = []

        for _ in range(num_to_add):
             new_particle, problem_after_add = initializing_optimizer_aspsogmpb(
                 min_coord, max_coord, dimension, problem_after_add)

             # Verifica mudança durante adição
             if problem_after_add.get('RecentChange', 0):
                  print("AVISO (Opt): Mudança detectada durante adição de partículas!")
                  # Adiciona as partículas novas criadas até agora + a última
                  optimizer['particle'].extend(added_particles_temp)
                  optimizer['particle'].append(new_particle)
                  particles = optimizer['particle'] # Atualiza atalho
                  # Recria espécies e retorna
                  species_list, optimizer = creating_species(optimizer)
                  return optimizer, problem_after_add, species_list

             added_particles_temp.append(new_particle)

        # Adiciona todas as novas partículas à população
        optimizer['particle'].extend(added_particles_temp)
        particles = optimizer['particle'] # Atualiza atalho

        # Recria espécies pois a população mudou
        print("INFO (Opt): Recriando espécies após adição de partículas.")
        species_list, optimizer = creating_species(optimizer)

    # Retorna o estado final da iteração
    return optimizer, problem, species_list


# =============================================
# Reação à Mudança Ambiental
# =============================================
def reaction_aspsogmpb(optimizer, problem, species_list_prev_env):
    """
    Executa a estratégia de reação do ASPSO a uma mudança ambiental.
    Equivalente a Reaction_ASPSO_GMPB.m.

    Args:
        optimizer (dict): Estado do otimizador.
        problem (dict): Estado do problema (já avançou Environmentcounter).
        species_list_prev_env (list): Espécies da iteração ANTES da mudança.

    Returns:
        tuple: (optimizer, problem)
            - optimizer (dict): Estado atualizado do otimizador.
            - problem (dict): Estado atualizado do problema.
    """
    print(f"--- Reagindo à mudança para o ambiente {problem['Environmentcounter']} (FE={problem['FE']}) ---")
    particles = optimizer.get('particle', [])
    if not particles:
        print("AVISO (React): Sem partículas para reagir.")
        return optimizer, problem

    dimension = optimizer['Dimension']
    min_coord = optimizer['MinCoordinate']
    max_coord = optimizer['MaxCoordinate']

    # --- 1. Identifica Trackers (da iteração anterior) ---
    tracker_indices_prev_env = [] # Índices na species_list_prev_env
    tracker_threshold = optimizer.get('teta', 1.0)
    for i, species in enumerate(species_list_prev_env):
         species_distance = species.get('distance', np.inf)
         if not np.isnan(species_distance) and species_distance < tracker_threshold:
              tracker_indices_prev_env.append(i)
    print(f"INFO (React): Identificados {len(tracker_indices_prev_env)} trackers do ambiente anterior.")

    # --- 2. Atualiza Estimativa de ShiftSeverity ---
    shift_distances = []
    valid_tracker_seeds_for_shift = [] # Guarda índices das sementes que tinham Pbest_past

    for tracker_idx in tracker_indices_prev_env:
        try:
             # Garante que o índice ainda é válido na lista de espécies anterior
             if tracker_idx >= len(species_list_prev_env): continue
             species_info = species_list_prev_env[tracker_idx]
             seed_idx = species_info['seed']
             # Garante que o índice da semente é válido na lista de partículas atual
             if seed_idx >= len(particles): continue
             seed_particle = particles[seed_idx]

             pbest_past = seed_particle.get('Pbest_past_environment')
             pbest_now = seed_particle.get('PbestPosition')

             if pbest_past is not None and pbest_now is not None:
                  # Calcula distância (shift)
                  dist = np.linalg.norm(pbest_past.flatten() - pbest_now.flatten())
                  shift_distances.append(dist)
                  seed_particle['Shifts'] = dist # Guarda na partícula semente
                  valid_tracker_seeds_for_shift.append(seed_idx) # Marca que esta semente contribuiu
             else:
                 # Limpa Shifts se não foi possível calcular
                 seed_particle['Shifts'] = None

             # Limpa Pbest_past independentemente de ter sido usado ou não
             seed_particle['Pbest_past_environment'] = None

        except (KeyError, IndexError, TypeError) as e:
             print(f"Aviso (React): Erro ao processar tracker {tracker_idx} para shift: {e}")
             continue

    # Atualiza ShiftSeverity se calculamos alguma distância
    if shift_distances:
        mean_shift = np.mean(shift_distances)
        # Adiciona um limite inferior pequeno para evitar shift zero ou negativo?
        mean_shift = max(mean_shift, 1e-6)
        print(f"INFO (React): Shift Severity estimado: {mean_shift:.4f} (baseado em {len(shift_distances)} trackers)")
        optimizer['ShiftSeverity'] = mean_shift
    else:
         # O que fazer se não calculou nenhum shift? Manter o anterior? Usar default=1?
         # O MATLAB parece manter o anterior implicitamente. Vamos manter.
         print(f"Aviso (React): Não foi possível calcular novas distâncias de shift. Mantendo ShiftSeverity={optimizer.get('ShiftSeverity', 1.0):.4f}")


    # --- 3. Aumenta Diversidade para Trackers (Reposicionamento) ---
    current_shift_severity = optimizer.get('ShiftSeverity', 1.0)
    print(f"INFO (React): Reposicionando membros dos trackers com shift_severity={current_shift_severity:.4f}")

    seeds_to_store_pbest = set() # Guarda sementes que foram processadas para salvar Pbest

    for tracker_idx in tracker_indices_prev_env:
        try:
             if tracker_idx >= len(species_list_prev_env): continue
             species_info = species_list_prev_env[tracker_idx]
             seed_idx = species_info['seed']
             if seed_idx >= len(particles): continue
             seed_particle = particles[seed_idx]

             # Pega Pbest da semente (que é o centro da nova dispersão)
             seed_pbest_pos = seed_particle['PbestPosition']
             if seed_pbest_pos is None: continue # Pula se não houver Pbest
             if seed_pbest_pos.ndim == 1: seed_pbest_pos = seed_pbest_pos.reshape(1,-1)

             # Reposiciona membros em torno da Pbest da semente
             for member_idx in species_info['member']:
                  if member_idx >= len(particles): continue # Membro inválido
                  if member_idx == seed_idx: continue # Não aplica shift aleatório na semente

                  R = np.random.randn(1, dimension)
                  norm_R = np.linalg.norm(R)
                  if norm_R > 1e-10:
                       shift_vec = (R / norm_R) * current_shift_severity
                  else:
                       shift_vec = np.zeros((1, dimension))

                  # Define a NOVA posição X do membro
                  particles[member_idx]['X'] = seed_pbest_pos + shift_vec

             # Reposiciona a semente exatamente na sua Pbest
             seed_particle['X'] = seed_pbest_pos.copy()

             # Marca para armazenar Pbest para a PRÓXIMA reação
             seeds_to_store_pbest.add(seed_idx)

        except (KeyError, IndexError) as e:
             print(f"Aviso (React): Erro ao reposicionar tracker {tracker_idx}: {e}")
             continue

    # Armazena Pbest_past_environment APENAS para as sementes processadas
    for idx in seeds_to_store_pbest:
         if idx < len(particles) and 'PbestPosition' in particles[idx]:
              particles[idx]['Pbest_past_environment'] = particles[idx]['PbestPosition'].copy()


    # --- 4. Tratamento de Limites (para TODAS as partículas) ---
    # Aplica após o reposicionamento dos trackers e antes da reavaliação
    print("INFO (React): Aplicando limites...")
    for i in range(len(particles)):
         try:
             particle = particles[i]
             # Garante que X existe antes de tentar acessá-lo
             if 'X' not in particle or particle['X'] is None: continue

             current_position = particle['X']
             if current_position.ndim == 1: current_position = current_position.reshape(1,-1)

             clamped_position = np.clip(current_position, min_coord, max_coord)
             clamped_mask = (current_position < min_coord) | (current_position > max_coord)

             # Zera velocidade correspondente SE o campo Velocity existir
             if 'Velocity' in particle and particle['Velocity'] is not None:
                  if particle['Velocity'].ndim == 1: particle['Velocity'] = particle['Velocity'].reshape(1,-1)
                  # Só zera se o shape for compatível
                  if particle['Velocity'].shape == clamped_mask.shape:
                       particle['Velocity'][clamped_mask] = 0.0
                  else:
                       # Fallback: Zera tudo se shapes não baterem (improvável)
                       particle['Velocity'] = np.zeros_like(clamped_position)


             # Atualiza posição X
             particle['X'] = clamped_position
         except (KeyError, IndexError, TypeError) as e:
             print(f"Aviso (React): Erro no bound handling da partícula {i}: {e}")
             continue

    # --- 5. Atualiza Memória (Pbest) para TODOS no Novo Ambiente ---
    print("INFO (React): Reavaliando e resetando Pbest de todas as partículas...")
    problem_after_eval = problem # Começa com o problema atual
    num_particles_react = len(particles)

    for i in range(num_particles_react):
         try:
             particle = particles[i]
             # Avalia fitness na posição X (pós-reação/limites) no NOVO ambiente
             # A flag RecentChange é 0 agora (resetada em main_aspsogmpb após chamar reaction)
             # Mas problem_after_eval ainda pode ter FE alto. fitness_aspsogmpb vai checar FE < MaxEvals.
             fitness_arr, problem_after_eval = fitness_aspsogmpb(particle['X'], problem_after_eval)
             current_fitness = fitness_arr[0, 0] if fitness_arr.size > 0 and not np.isnan(fitness_arr[0,0]) else -np.inf

             # Reseta Pbest para o estado atual
             particle['FitnessValue'] = current_fitness
             particle['PbestFitness'] = current_fitness
             # Garante que PbestPosition exista antes de copiar X
             if 'PbestPosition' not in particle:
                  particle['PbestPosition'] = np.zeros_like(particle['X']) # Ou outra inicialização
             particle['PbestPosition'] = particle['X'].copy()

             # Se a avaliação causar outra mudança, problem_after_eval terá RecentChange=1
             # Isso será pego no próximo loop principal em main_aspsogmpb

         except (KeyError, IndexError, TypeError) as e:
             print(f"Aviso (React): Erro ao reavaliar/resetar partícula {i}: {e}")
             # Define um Pbest inválido para sinalizar problema?
             particle['FitnessValue'] = -np.inf
             particle['PbestFitness'] = -np.inf
             continue

    problem = problem_after_eval # Atualiza o problema com os FEs consumidos

    # --- 6. Atualiza Parâmetros/Limiares do Otimizador ---
    print("INFO (React): Atualizando parâmetros adaptativos...")
    current_shift_severity = optimizer.get('ShiftSeverity', 1.0) # Pega o valor atualizado (ou default)
    optimizer['MaxDeactivation'] = optimizer.get('rho', 0.7) * current_shift_severity
    optimizer['CurrentDeactivation'] = optimizer['MaxDeactivation'] # Reseta para o máximo
    optimizer['teta'] = current_shift_severity # Limiar do tracker = shift estimado
    optimizer['beta'] = 1.0 # Reseta fator de decaimento da desativação

    print(f"--- Reação concluída (FE={problem['FE']}) ---")
    return optimizer, problem