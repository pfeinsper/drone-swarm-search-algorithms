from DSSE import CoverageDroneSwarmSearch
import pandas as pd


def expanding_search_4_agent(direcao1, direcao2, direcao3, direcao4, agents):

    actions = {}

    for agent in agents:
        if agent == "drone0":
            if direcao1 == "cima":
                actions[agent] = 2
            elif direcao1 == "direita":
                actions[agent] = 1
            elif direcao1 == "baixo":
                actions[agent] = 3
            elif direcao1 == "esquerda":
                actions[agent] = 0
        elif agent == "drone1":
            if direcao2 == "cima":
                actions[agent] = 2
            elif direcao2 == "direita":
                actions[agent] = 1
            elif direcao2 == "baixo":
                actions[agent] = 3
            elif direcao2 == "esquerda":
                actions[agent] = 0
        elif agent == "drone2":
            if direcao3 == "cima":
                actions[agent] = 2
            elif direcao3 == "direita":
                actions[agent] = 1
            elif direcao3 == "baixo":
                actions[agent] = 3
            elif direcao3 == "esquerda":
                actions[agent] = 0
        elif agent == "drone3":
            if direcao4 == "cima":
                actions[agent] = 2
            elif direcao4 == "direita":
                actions[agent] = 1
            elif direcao4 == "baixo":
                actions[agent] = 3
            elif direcao4 == "esquerda":
                actions[agent] = 0

    return actions


def main():
    env = CoverageDroneSwarmSearch(
        drone_amount=4,
        render_mode="human",
        prob_matrix_path='min_matrix.npy',
        timestep_limit=200
    )

    opt = {
        "drones_positions": [(2, 2), (2, 6), (6, 6), (6, 2)],
    }
    
    observations, info = env.reset(options=opt)

    step = 0
    infos_list = []
    direcoes1 = ["direita", "baixo", "esquerda", "cima"]
    direcoes2 = ["esquerda", "cima", "direita", "baixo"]
    direcoes3 = ["cima", "direita", "baixo", "esquerda"]
    direcoes4 = ["baixo", "esquerda", "cima", "direita"]
    passos_inciais = 1
    passos = 1
    contador = 1
    indice_dir = 0

    while env.agents:
        if passos <= 0:
            passos = passos_inciais

            if contador <= 0:
                contador = 1
                passos_inciais += 1
            else:
                contador -= 1

            if indice_dir >= 3:
                indice_dir = 0
            else:
                indice_dir += 1

        step += 1
        actions = expanding_search_4_agent(direcoes1[indice_dir], direcoes2[indice_dir], direcoes3[indice_dir], direcoes4[indice_dir], env.agents)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        info = infos['drone0']
        #print(observations['drone0'][0])
        info['step'] = step
        infos_list.append(info)
        print(info)
        passos -= 1

    df = pd.DataFrame(infos_list)
    df.to_csv('results/expanding_search_4_agent.csv', index=False)

if __name__ == "__main__":
    main()
