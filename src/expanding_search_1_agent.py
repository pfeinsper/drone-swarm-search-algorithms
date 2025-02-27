from DSSE import CoverageDroneSwarmSearch
import pandas as pd


def expanding_search_single_agent(direcao, agents):

    actions = {}

    for agent in agents:

        if direcao == "cima":
            actions[agent] = 2
        elif direcao == "direita":
            actions[agent] = 1
        elif direcao == "baixo":
            actions[agent] = 3
        elif direcao == "esquerda":
            actions[agent] = 0

    return actions


def main():
    env = CoverageDroneSwarmSearch(
        drone_amount=1,
        render_mode="human",
        prob_matrix_path='min_matrix.npy',
        timestep_limit=200
    )

    opt = {
        "drones_positions": [(4, 4)],
    }
    
    observations, info = env.reset(options=opt)

    step = 0
    infos_list = []
    direcoes = ["cima", "direita", "baixo", "esquerda"]
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
        actions = expanding_search_single_agent(direcoes[indice_dir], env.agents)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        info = infos['drone0']
        #print(observations['drone0'][0])
        info['step'] = step
        infos_list.append(info)
        print(info)
        passos -= 1

    df = pd.DataFrame(infos_list)
    df.to_csv('results/expanding_search_1_agent.csv', index=False)

if __name__ == "__main__":
    main()
