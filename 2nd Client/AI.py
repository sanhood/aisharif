import Model
from random import randint
import Agent
import numpy as np
import keras
import os
# import tensorflow as tf
class AI:
    turn = 0
    old_world = None
    def preprocess(self, world):
        print("preprocess")
        # graph = tf.get_default_graph()
        # global graph
        # with graph.as_default():
        self.moveAgent = Agent.DQNAgent("move", world)
        if os.path.exists(self.moveAgent.name):
            with keras.backend.get_session().graph.as_default():
                self.moveAgent.load()
                print("loaded!")
        # self.moveAgent.model._make_predict_function()

    def pick(self, world):
        print("pick")
        hero_names = [hero_name for hero_name in Model.HeroName]
        world.pick_hero(hero_names[0])

    def move(self, world):
        with keras.backend.get_session().graph.as_default():
            if self.old_world is not None:
                moves = self.evaluate(self.old_world,world)
                for (state, action, reward, next_state) in moves:
                    self.moveAgent.remember(state, action, reward, next_state)
                    # print("loc: {},{}".format(state.hero.current_cell.row, next_state.hero.current_cell.row))
                    if len(self.moveAgent.memory) > self.moveAgent.batch_size:
                        self.moveAgent.replay()

            print("move")
            dirs = [direction for direction in Model.Direction]
            for hero in world.my_heroes:
                state = State(hero,world)
                reshaped_state = np.reshape(state.state,[1,len(world.map.cells)**2])
                action = self.moveAgent.act(reshaped_state)
                # print(action)
                move = dirs[action]
                world.move_hero(hero=hero, direction=move)
                # next_state, reward = self.evaluate(hero,world)
                # self.previous_moves.append((state, action, reward, next_state))
                # if e % 50 == 0:
                #     agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
                # world.move_hero(hero=hero, direction=move)
            self.old_world = world
        # for hero in world.my_heroes:
        #     world.move_hero(hero=hero, direction=dirs[randint(0, len(dirs) - 1)])

    def action(self, world):
        print("action")
        str = ""
        for hero in world.my_heroes:
            row_num = randint(0, world.map.row_num)
            col_num = randint(0, world.map.column_num)
            abilities = hero.abilities
            world.cast_ability(hero=hero, ability=abilities[randint(0, len(abilities) - 1)],
                               cell=world.map.get_cell(row_num, col_num))
        print(self.turn)
        print(world.game_constants.max_turns)
        if world.game_constants.max_turns+1 == self.turn:
            print("saved!")
            self.moveAgent.save()
        self.turn += 1
        # for i in range(len(cells)):
        #     for j in range(len(cells[i])):
        #         str = str + "{},{}  ".format(int(cells[i][j].is_wall),int(cells[i][j].is_in_vision))
        #     str = str + "\n"
        # str = str + "\n-----------------------------------\n"
        # with open("Output.txt", "a") as text_file:
        #     text_file.write(str)
    # def evaluate(self,hero,world):
    #     reward = 0
    #     if hero.current_cell.is_in_objective_zone:
    #         reward = 100
    #     next_state = State(hero,world)
    #     return next_state,reward

    def evaluate(self,old_world,new_world):
        old_hero = old_world.my_heroes
        new_hero = new_world.my_heroes
        moves = []
        for index,hero in enumerate(old_hero):
            state = State(hero,old_world)
            next_state = State(new_hero[index],new_world)
            action = None
            if hero.current_cell.row +1 == new_hero[index].current_cell.row:
                action = 1
            elif hero.current_cell.row -1 == new_hero[index].current_cell.row:
                action = 0
            elif hero.current_cell.column +1 == new_hero[index].current_cell.column:
                action = 2
            elif hero.current_cell.column -1 == new_hero[index].current_cell.column:
                action = 3
            # print("old: {},{}  new:{},{}".format(hero.current_cell.row,hero.current_cell.column,new_hero[index].current_cell.row,new_hero[index].current_cell.column))
            reward = self.get_reward(new_hero[index],new_world)
            # if new_hero[index].current_cell.is_in_objective_zone:
            #     reward = 100
            #     print("rewarded!!!!!")
            if action is not None:
                moves.append((state, action, reward, next_state))

        return moves

    def get_reward(self,hero,world):
        reward = 0
        objective_zone_cell = world.map.objective_zone[0]
        reward = 2 * (30 - world.manhattan_distance(hero.current_cell,objective_zone_cell))
        if hero.current_cell.is_in_objective_zone:
            reward = 200
            print("rewarded 100!!!!!")
        return reward
class State:

    def __init__(self,hero,world):
        self.current_cell = hero.current_cell
        self.state = np.zeros((len(world.map.cells),len(world.map.cells)))
        self.hero = hero
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


