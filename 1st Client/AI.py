import Model
from random import randint
import Agent
import numpy as np
class AI:
    def preprocess(self, world):
        print("preprocess")
        # graph = tf.get_default_graph()
        # global graph
        # with graph.as_default():
        # self.moveAgent = Agent.DQNAgent("move",world)
        # self.moveAgent.save()
        # first_layer_weights = self.moveAgent.layers[0].get_weights()[0]
        # print(first_layer_weights)
        # if not os.path.exists("{}".format(self.moveAgent.name)):
        #     with keras.backend.get_session().graph.as_default():
        #         self.moveAgent.load()
        #         print("******************")
        #         first_layer_weights = self.moveAgent.layers[0].get_weights()[0]
        #         print(first_layer_weights)
        # self.moveAgent.model._make_predict_function()

    def pick(self, world):
        print("pick")
        hero_names = [hero_name for hero_name in Model.HeroName]
        world.pick_hero(hero_names[randint(0, len(hero_names) - 1)])

    def move(self, world):
        print("move")
        # dirs = [direction for direction in Model.Direction]
        # for hero in world.opp_heroes:
        #     state = State(hero,world)
        #     reshaped_state = np.reshape(state.state,[1,len(world.map.cells)**2])
        #     action = self.moveAgent.act(reshaped_state)
        #     move = dirs[np.argmax(action)]
        #     world.move_hero(hero=hero, direction=move)
        #     next_state, reward = self.evaluate(hero,world)
        #     self.moveAgent.remember(state, action, reward, next_state)
        #     if len(self.moveAgent.memory) > self.moveAgent.batch_size:
        #         self.moveAgent.replay()
            # if e % 50 == 0:
            #     agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
            # world.move_hero(hero=hero, direction=move)


        # for hero in world.my_heroes:
        #     world.move_hero(hero=hero, direction=dirs[randint(0, len(dirs) - 1)])
        dirs = [direction for direction in Model.Direction]
        for hero in world.my_heroes:
            world.move_hero(hero=hero, direction=dirs[randint(0, len(dirs) - 1)])
            # print("loc: {},{}".format(hero.current_cell.row, hero.current_cell.column))

    def action(self, world):
        print("action")
        str = ""
        for hero in world.my_heroes:
            row_num = randint(0, world.map.row_num)
            col_num = randint(0, world.map.column_num)
            abilities = hero.abilities
            world.cast_ability(hero=hero, ability=abilities[randint(0, len(abilities) - 1)],
                               cell=world.map.get_cell(row_num, col_num))
        # for i in range(len(cells)):
        #     for j in range(len(cells[i])):
        #         str = str + "{},{}  ".format(int(cells[i][j].is_wall),int(cells[i][j].is_in_vision))
        #     str = str + "\n"
        # str = str + "\n-----------------------------------\n"
        # with open("Output.txt", "a") as text_file:
        #     text_file.write(str)
    def evaluate(self,hero,world):
        reward = 0
        if hero.current_cell.is_in_objective_zone:
            reward = 100
            print("it reached the objective zone")
        next_state = State(hero,world)
        return next_state,reward

class State:

    def __init__(self,Hero,world):
        self.current_cell = Hero.current_cell
        self.state = np.zeros((len(world.map.cells),len(world.map.cells)))
        for i in range(len(world.map.cells)):
            for j in range(len(world.map.cells[i])):
                if not world.map.cells[i][j].is_in_vision:
                    self.state[i][j] = 0
                elif world.map.cells[i][j].is_wall:
                    self.state[i][j] = -2
                else:
                    self.state[i][j] = 3
        for hero in world.opp_heroes:
            self.state[hero.current_cell.row][hero.current_cell.column] = 1
        # for enemy_hero in world.opp_heroes:
        #     self.state[enemy_hero.current_cell.row][enemy_hero.current_cell.column] = -1
        self.state[self.current_cell.row][self.current_cell.column] = 2


