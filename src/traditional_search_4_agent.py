from DSSE import CoverageDroneSwarmSearch
import pandas as pd
import random

def traditional_search_4_agent(obs, agents, opt, passos):

    actions = {}
    
    for agent in agents:
        if agent == "drone0":
            start_position = (opt['drones_positions'][0][0], opt['drones_positions'][0][1])

            if (obs[agent][0][0] - start_position[0]) % 2 == 0:
                if passos == 0:
                    actions[agent] = 0
                else:
                    actions[agent] = 3
            elif (obs[agent][0][0] - start_position[0]) % 2 != 0:
                if passos == 0:
                    actions[agent] = 0
                else:
                    actions[agent] = 2
        elif agent == "drone1":
            start_position = (opt['drones_positions'][1][0], opt['drones_positions'][1][1])
            
            if (obs[agent][0][0] - start_position[0]) % 2 == 0:
                if passos == 0:
                    actions[agent] = 1
                else:
                    actions[agent] = 3
            elif (obs[agent][0][0] - start_position[0]) % 2 != 0:
                if passos == 0:
                    actions[agent] = 1
                else:
                    actions[agent] = 2
        elif agent == "drone2":
            start_position = (opt['drones_positions'][2][0], opt['drones_positions'][2][1])

            if (obs[agent][0][0] - start_position[0]) % 2 == 0:
                if passos == 0:
                    actions[agent] = 0
                else:
                    actions[agent] = 2
            elif (obs[agent][0][0] - start_position[0]) % 2 != 0:
                if passos == 0:
                    actions[agent] = 0
                else:
                    actions[agent] = 3
        elif agent == "drone3":
            start_position = (opt['drones_positions'][3][0], opt['drones_positions'][3][1])
            
            if (obs[agent][0][0] - start_position[0]) % 2 == 0:
                if passos == 0:
                    actions[agent] = 1
                else:
                    actions[agent] = 2
            elif (obs[agent][0][0] - start_position[0]) % 2 != 0:
                if passos == 0:
                    actions[agent] = 1
                else:
                    actions[agent] = 3

    return actions


def main():

    infos_list = []

    for i in range(25):

        env = CoverageDroneSwarmSearch(
            drone_amount=4,
            render_mode="human",
            prob_matrix_path='min_matrix.npy',
            timestep_limit=200
        )

        r_l = random.randint(-1, 1)
        r_c = random.randint(-1, 1)

        opt = {
            "drones_positions": [(4+r_l, 3+r_c), (5+r_l, 3+r_c), (4+r_l, 4+r_c), (5+r_l, 4+r_c)],
        }
        observations, info = env.reset(options=opt)

        step = 0

        passos = 13

        while env.agents:
            if passos < 0:
                passos = 12
            
            step += 1
            actions = traditional_search_4_agent(observations, env.agents, opt, passos)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            info = infos['drone0']
            #print(observations['drone0'][0])
            info['step'] = step
            infos_list.append(info)
            print(info)
            passos -= 1

    df = pd.DataFrame(infos_list)
    df.to_csv('results/traditional_search_4_agent.csv', index=False)

if __name__ == "__main__":
    main()