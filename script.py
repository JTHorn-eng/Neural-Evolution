import math
from random import random, randint, choice, sample
import numpy as np
import gymnasium as gym
import copy
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import matplotlib.ticker as mticker
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


class Config:

    ADD_NODE_PROB = 0.1
    ADD_CONN_PROB = 0.1
    CHG_WEIGHT_PROB = 0.8
    NUM_INPUT_NODES = 4
    NUM_OUTPUT_NODES = 1
    MAX_POPULATION = 50
    MAX_GENERATIONS = 100
    SIMULATION = "CartPole-v1"


class Link:

    def __init__(self, inp, out, weight=0.0, enabled=True):
        self.weight = randint(1, 10) / 10
        self.inp = inp
        self.out = out
        self.enabled = enabled

    def __str__(self):
        return f"{self.inp} --- {self.weight} --->> {self.out} | {self.enabled}"

    def __eq__(self, con):
        return all([self.inp == con.inp, self.out == con.out])


class Node:

    def __init__(self, id, typ, val=0.0, links=[]):
        self.id = id
        self.val = val
        self.typ = typ
        self.links = links

    def sigmoid(self):
        try:
            return 1 / (1 + math.exp(-self.val))
        except OverflowError:
            return 0.0

    def __str__(self):
        return f"{self.id} | Value: {self.val} | Type: {self.typ}"

    def __eq__(self, node):

        links_chk = False
        if len(node.links) == len(self.links):
            for x in range(len(node.links)):
                if node.links[x] != self.links[x]:
                    links_chk = True
                else:
                    links_chk = False

        return all(
            [
                node.id == self.id,
                node.typ == self.typ,
                links_chk,
            ]
        )


class Genome:

    GLOBAL_ID = 0
    generation = 0

    def __init__(self, inputs_num, outputs_num):
        self.fitness = 0.0
        self.node_dict = dict()
        self.name = ""
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        self.create_new(self.inputs_num, self.outputs_num)
        self.id = Genome.GLOBAL_ID
        Genome.GLOBAL_ID += 1

    def __len__(self):
        return len(self.node_dict)

    def create_new(self, inputs_num=1, outputs_num=2):

        self.name = f"{Genome.generation}"

        self.node_dict = {
            float(k): Node(id=float(k), typ="input") for k in range(inputs_num)
        }
        self.node_dict.update(
            {
                float(k): Node(id=float(k), typ="output")
                for k in range(inputs_num, inputs_num + outputs_num)
            }
        )
        output_nodes = list(
            filter(lambda x: x.typ == "output", self.node_dict.values())
        )

        for node in self.node_dict.values():
            node.links = [
                Link(inp=node.id, out=n.id)
                for n in output_nodes
                if (node.id < n.id) and node.typ != n.typ
            ]

    def feed_forward(self, inputs):
        # Reset all node values
        for node in self.node_dict.values():
            node.val = 0.0

        # Set inputs
        for node in self.node_dict.values():
            if node.id in inputs:
                node.val = inputs[node.id]

        for node in sorted(self.node_dict.values(), key=lambda x: x.id):
            node.val = node.sigmoid()
            for link in list(
                filter(lambda n: (n.out and n.enabled) in self.node_dict, node.links)
            ):
                if link.out in self.node_dict:
                    self.node_dict[link.out].val += link.weight * node.val

        return list(filter(lambda x: x.typ == "output", self.node_dict.values()))[0].val

    def mutate(self):

        # Mutate weights
        if random() <= Config.CHG_WEIGHT_PROB:
            for node in self.node_dict.values():
                for x in node.links:
                    if random() < 0.2:
                        x.weight += np.random.normal(loc=0.0, scale=1.0)
                        x.weight = max(min(x.weight, 1.0), -1.0)

        # Add connection
        if random() <= Config.ADD_CONN_PROB:
            from_node = choice(
                list(
                    filter(
                        lambda x: x.typ == "input" or x.typ == "hidden",
                        self.node_dict.values(),
                    )
                )
            )
            to_node = choice(
                list(
                    filter(
                        lambda x: (x.id > from_node.id)
                        and (x.typ == "hidden" or x.typ == "output"),
                        self.node_dict.values(),
                    )
                )
            )

            chk = True
            for link in from_node.links:
                if link.inp == from_node.id and link.out == to_node.id and link.enabled:
                    chk = False
                    break
            if chk:
                from_node.links.append(
                    Link(
                        inp=from_node.id,
                        out=to_node.id,
                        weight=np.random.normal(loc=0.0, scale=1.0),
                    )
                )

        # Add node
        if random() <= Config.ADD_NODE_PROB:
            from_node = choice(
                list(
                    filter(
                        lambda x: x.typ == "input" or x.typ == "hidden",
                        self.node_dict.values(),
                    )
                )
            )
            to_node = choice(
                list(
                    filter(
                        lambda x: (x.id > from_node.id)
                        and (x.typ == "hidden" or x.typ == "output"),
                        self.node_dict.values(),
                    )
                )
            )

            # Delete old connection
            for link in from_node.links:
                if link.inp == from_node.id and link.out == to_node.id:
                    link.enabled = False
                    break

            # Add new node
            new_id = (
                ((from_node.id + to_node.id) / 2)
                if not ((from_node.id + to_node.id) / 2) in self.node_dict
                else (0.1 + ((from_node.id + to_node.id) / 2))
            )
            new_node = Node(
                id=new_id, typ="hidden", links=[Link(inp=new_id, out=to_node.id)]
            )
            from_node.links.append(Link(inp=from_node.id, out=new_id))
            self.node_dict[new_id] = new_node

    def copy(self):

        gen = Genome(self.inputs_num, self.outputs_num)
        gen.node_dict = copy.deepcopy(self.node_dict)
        gen.fitness = 0
        gen.name = self.name
        return gen

    @staticmethod
    def crossover(p1, p2):

        if p1.fitness > p2.fitness:
            return p1.copy()
        else:
            return p2.copy()

    def __str__(self):

        links_ls = list()
        for n in self.node_dict.values():
            for l in n.links:
                links_ls.append(l)

        return (
            f"Network {self.id}\n\nNodes:\n"
            + "\n".join([str(n) for n in list(self.node_dict.values())])
            + "\n\nConnections:\n"
            + "\n".join([str(l) for l in links_ls])
        )


class NEAT:

    def __init__(self):

        self.generation = 0
        self.inputs_num = Config.NUM_INPUT_NODES
        self.outputs_num = Config.NUM_OUTPUT_NODES
        self.max_population = Config.MAX_POPULATION
        self.population = [
            Genome(inputs_num=self.inputs_num, outputs_num=self.outputs_num)
        ] * 3
        self.thresh = -1

    def run_evolution(self):

        # Select
        new_population = list()
        sorted_genomes = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        for g in range(0, min(len(sorted_genomes), 10)):
            new_population.append(sorted_genomes[g])

        self.population = new_population

        # Cross and mutate
        while len(self.population) < self.max_population:
            p1 = choice(self.population)
            p2 = choice(self.population)

            asd = Genome.crossover(p1, p2)
            asd.mutate()

            self.population.append(asd)


class Simulation:

    def __init__(self):
        self.env = gym.make(Config.SIMULATION, render_mode="rgb_array")
        self.obvs = self.env.reset()

    def play(self, gene, polecart_vid_num=0, record=False):

        if record:
            self.env = gym.make(Config.SIMULATION, render_mode="rgb_array")
            self.obvs = self.env.reset()

            self.env = gym.wrappers.RecordVideo(
                env=self.env,
                video_folder="./cart_pole_videos",
                name_prefix=f"{Config.SIMULATION}{polecart_vid_num}",
                episode_trigger=lambda x: x,
            )
            self.env.start_video_recorder()
        else:
            self.env = gym.make(Config.SIMULATION, render_mode="rgb_array")

        action = 0
        done = False
        temp_reward = 0
        observation = self.env.reset()

        for r in range(5000):
            observation, reward, done, _, _ = self.env.step(action)

            if done:  # If the simulation was over last iteration, exit loop
                break

            observation_dict = {
                idx: observation[idx] for idx, m in enumerate(observation)
            }

            # Pick a move left/right action according to network output
            output = gene.feed_forward(inputs=observation_dict)
            action = 1 if output > 0.5 else 0

            # Make the action, record reward
            temp_reward += reward

        gene.fitness = temp_reward

        if record:
            self.env.close_video_recorder()
            self.env.close()


sim = Simulation()
neat = NEAT()
generations = Config.MAX_GENERATIONS


def run(max_population_size=50):

    mean_scores = []
    best_gene = None

    gen_count = 0
    best_genes = []

    neat.max_population = Config.MAX_POPULATION

    for x in range(generations):
        print(f"Generation Number: {gen_count}")
        neat.run_evolution()

        # Reset fitness scores
        # Reset node values

        actions_rec = None
        for g in neat.population:
            sim.play(gene=g)

        sorted_population = sorted(
            neat.population, key=lambda x: x.fitness, reverse=True
        )

        best_gene = sorted_population[0].copy()

        mean_scores.append(np.mean(list(map(lambda x: x.fitness, neat.population))))
        print(f"Mean Fitness: {np.mean([i.fitness for i in neat.population])}")

        best_genes.append(best_gene)

        gen_count += 1

    return best_genes, mean_scores


best_genes, mean_scores = run(max_population_size=Config.MAX_POPULATION)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(range(0, generations)), y=mean_scores, mode="lines", name="lines")
)
# Edit the layout
fig.update_layout(
    title="Mean fitness of genomes during training",
    xaxis_title="Generation Number",
    yaxis_title="Mean Fitness",
)

fig.show()

# print("\n\nRunning the best genome\n\n")
# sim.play(gene=best_genes[1], polecart_vid_num=0, record=True)
# sim.play(gene=best_genes[15], polecart_vid_num=1, record=True)
# sim.play(gene=best_genes[30], polecart_vid_num=2, record=True)
