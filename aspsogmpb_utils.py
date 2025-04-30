# aspsogmpb_utils.py
import numpy as np
# Removido: from scipy.spatial.distance import cdist # Usaremos np.linalg.norm

# Importa a função de fitness (necessária para inicializar partícula)
try:
    from fitness_eval import fitness_aspsogmpb
except ImportError:
    print("AVISO: fitness_eval.py não encontrado em aspsogmpb_utils.py.")
    # Define uma função placeholder se não encontrar
    def fitness_aspsogmpb(X, problem):
        print("AVISO: Usando função fitness_aspsogmpb placeholder em utils!")
        # Retorna um fitness fixo e o problema inalterado para permitir testes básicos
        return np.array([[0.0]] * X.shape[0]), problem

# =============================================
# Inicialização de Partícula
# =============================================
def initializing_optimizer_aspsogmpb(lb, ub, dimension, problem):
    """
    Inicializa uma única partícula para o otimizador ASPSO.
    Equivalente a InitializingOptimizer_ASPSO_GMPB.m (assumindo npop=1).

    Args:
        lb (float): Limite inferior das coordenadas.
        ub (float): Limite superior das coordenadas.
        dimension (int): Dimensionalidade.
        problem (dict): Dicionário do estado do problema.

    Returns:
        tuple: (particle_state, problem)
            - particle_state (dict): Dicionário representando a partícula inicializada.
            - problem (dict): Dicionário do problema potencialmente atualizado pela chamada de fitness.
    """
    particle_state = {}

    # Posição Inicial (1 partícula, shape (1, dimension))
    particle_state['X'] = lb + (ub - lb) * np.random.rand(1, dimension)

    # Pbest Inicial
    particle_state['PbestPosition'] = particle_state['X'].copy() # Usa cópia

    # Velocidade Inicial
    particle_state['Velocity'] = np.zeros((1, dimension))

    # Calcula Fitness Inicial (chama a função importada)
    # fitness_aspsogmpb espera X como (n, dim) e retorna (fitness_array (n,1), updated_problem)
    fitness_value_arr, problem = fitness_aspsogmpb(particle_state['X'], problem)

    # Extrai valor escalar de fitness; lida com possível NaN retornado
    initial_fitness = fitness_value_arr[0, 0] if fitness_value_arr.size > 0 else np.nan

    # Define FitnessValue e PbestFitness baseado na flag RecentChange
    # Se RecentChange for 1, invalida Pbest e Fitness (define como -inf)
    if problem.get('RecentChange', 0):
        invalid_fitness = -np.inf
        particle_state['FitnessValue'] = invalid_fitness
        particle_state['PbestFitness'] = invalid_fitness
    else:
        # Garante que não estamos atribuindo NaN
        particle_state['FitnessValue'] = initial_fitness if not np.isnan(initial_fitness) else -np.inf
        particle_state['PbestFitness'] = initial_fitness if not np.isnan(initial_fitness) else -np.inf


    # Outros Campos Essenciais
    particle_state['Processed'] = False # Flag para creating_species (Booleano é mais Pythonico)
    particle_state['Shifts'] = None     # Usado em Reaction (inicia como None)
    particle_state['Pbest_past_environment'] = None # Usado em Reaction (inicia como None)

    return particle_state, problem


# =============================================
# Criação de Espécies
# =============================================
def creating_species(optimizer):
    """
    Agrupa partículas em espécies baseado em PbestFitness e proximidade (PbestPosition).
    Equivalente a CreatingSpecies.m.

    Args:
        optimizer (dict): Dicionário do estado do otimizador, deve conter:
                          'particle': lista de dicionários de partículas.
                          'SwarmMember': tamanho desejado para cada espécie (int).

    Returns:
        tuple: (species_list, optimizer)
            - species_list (list): Lista de dicionários, cada um representando uma espécie.
                                   {'seed':int, 'member':list[int], 'remove':int, 'Active':int, 'distance':float}
            - optimizer (dict): Otimizador atualizado (campo 'Processed' das partículas modificado).
    """
    # Verifica se há partículas
    if 'particle' not in optimizer or not optimizer['particle']:
         return [], optimizer # Retorna lista vazia se não há partículas

    num_particles = len(optimizer['particle'])
    particles = optimizer['particle'] # Atalho para a lista de partículas

    # 1. Inicializa flag 'Processed' em todas as partículas
    for i in range(num_particles):
        particles[i]['Processed'] = False

    # 2. Ordena partículas por PbestFitness descendente
    try:
        # Extrai fitness, tratando possíveis NaNs ou Infs para ordenação segura
        fitnesses = np.array([p.get('PbestFitness', -np.inf) for p in particles])
        fitnesses = np.nan_to_num(fitnesses, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        # Obtém índices que ordenariam fitnesses (ascendente) e inverte para descendente
        sort_index = np.argsort(fitnesses)[::-1]
    except KeyError:
        raise KeyError("Falha ao criar espécies: Campo 'PbestFitness' não encontrado consistentemente nas partículas.")

    species_list = []
    swarm_member_count = optimizer.get('SwarmMember', 1) # Pega do otimizador, default 1

    # 3. Itera sobre os índices ordenados para formar espécies
    for jj in range(num_particles):
        seed_idx = sort_index[jj] # Índice da partícula candidata a semente

        # Se a partícula ainda não foi processada, ela inicia uma nova espécie
        if not particles[seed_idx]['Processed']:
            # Inicializa a nova espécie
            current_species = {
                'seed': seed_idx,         # Índice da partícula semente
                'member': [seed_idx],     # Lista de índices dos membros (começa com a semente)
                'remove': 0,              # Flag para exclusão (0 = não remover)
                'Active': 1,              # Flag de atividade (1 = ativa)
                'distance': 0.0           # Raio da espécie (distância semente -> membro mais distante)
            }
            particles[seed_idx]['Processed'] = True # Marca semente como processada

            try:
                seed_pbest_pos = particles[seed_idx]['PbestPosition'].flatten() # Garante 1D para norm
            except KeyError:
                 print(f"Aviso: PbestPosition faltando na partícula semente {seed_idx}. Pulando espécie.")
                 continue # Não pode formar espécie sem posição da semente

            # Se o tamanho desejado da espécie for > 1, procura membros
            if swarm_member_count > 1:
                # Calcula distâncias da semente para outras partículas NÃO processadas
                distances = []
                unprocessed_indices = []
                for ii in range(num_particles):
                    if not particles[ii]['Processed']:
                        try:
                            other_pbest_pos = particles[ii]['PbestPosition'].flatten()
                            # Calcula distância Euclidiana usando numpy.linalg.norm
                            dist = np.linalg.norm(seed_pbest_pos - other_pbest_pos)
                            distances.append(dist)
                            unprocessed_indices.append(ii) # Guarda índice original da partícula
                        except KeyError:
                            # Pula partícula se não tiver PbestPosition
                            print(f"Aviso: PbestPosition faltando na partícula {ii} ao calcular distâncias para semente {seed_idx}.")
                            continue
                        except Exception as e:
                             print(f"Erro calculando distância entre semente {seed_idx} e partícula {ii}: {e}")
                             continue


                # Se encontrou partículas não processadas para potencialmente adicionar
                if distances:
                    distances = np.array(distances)
                    unprocessed_indices = np.array(unprocessed_indices)

                    # Ordena por distância (ascendente) e pega os índices relativos
                    sorted_dist_indices_rel = np.argsort(distances)

                    # Número de membros a adicionar (sem contar a semente)
                    num_to_add = min(swarm_member_count - 1, len(sorted_dist_indices_rel))

                    max_dist_in_species = 0.0 # Guarda a distância do membro mais distante adicionado

                    # Adiciona os membros mais próximos
                    for n in range(num_to_add):
                        # Índice relativo -> índice na lista 'unprocessed_indices' -> índice original da partícula
                        member_original_idx = unprocessed_indices[sorted_dist_indices_rel[n]]
                        current_species['member'].append(member_original_idx)
                        particles[member_original_idx]['Processed'] = True # Marca membro como processado

                        # Atualiza a distância máxima (raio) se este membro for o mais distante até agora
                        current_member_distance = distances[sorted_dist_indices_rel[n]]
                        if current_member_distance > max_dist_in_species:
                             max_dist_in_species = current_member_distance

                    # Define a distância final da espécie
                    current_species['distance'] = max_dist_in_species
                # Se não havia outras partículas não processadas, a distância permanece 0 (espécie só com semente)

            # Adiciona a espécie completa à lista
            species_list.append(current_species)

    # Retorna a lista de espécies e o otimizador com as flags 'Processed' atualizadas
    return species_list, optimizer