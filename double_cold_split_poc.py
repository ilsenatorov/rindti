import os.path
import sys
import networkx as nx
import numpy as np
from copy import copy
import pandas as pd
from pygad import GA

params = {
    "GA": {
        "G": 1000,  # number of generations in optimization
        "P": 6,  # number of parents
        "KP": 1,  # keep parents to next generation
        "CP": 0.8,  # crossover probability
        "MP": 0.8,  # mutation probability
        "D": 0.1,  # weight of drug balance between train and test sets
        "B": 3,  # weight of protein balance between train and test sets
    }
}


def show_data_split(split, dataset):
    data = pd.read_csv(dataset, sep="\t")
    prots = data["Target_ID"].unique()
    drugs = data["Drug_ID"].unique()

    train_solution = [bool(x) for x in split]
    test_solution = [not bool(x) for x in split]

    train_prots = prots[train_solution[:len(prots)]]
    test_prots = prots[test_solution[:len(prots)]]
    train_drugs = drugs[train_solution[len(prots):]]
    test_drugs = drugs[test_solution[len(prots):]]

    train_data = data[data["Target_ID"].isin(train_prots) & data["Drug_ID"].isin(train_drugs)]
    test_data = data[data["Target_ID"].isin(test_prots) & data["Drug_ID"].isin(test_drugs)]

    print(
        f"=================================================\n"
        f"Final Split Evaluation\n"
        f"-------------------------------------------------\n"
        f"Total number of interactions: {len(data)}\n"
        f"Dropped number of interactions: {len(data) - len(train_data) - len(test_data)} "
        f"({(len(data) - len(train_data) - len(test_data)) / len(data):.2})\n"
        f"Total number of proteins: {len(prots)}\n"
        f"Total number of drugs: {len(drugs)}\n"
        f"-------------------------------------------------\n"
        f"Number of interactions in training: {len(train_data)} ({len(train_data) / len(data):.2})\n"
        f"Number of interactions in testing: {len(test_data)} ({len(test_data) / len(data):.2})\n"
        f"Fraction of interactions in training: {len(train_data) / (len(train_data) + len(test_data)):.2}\n"
        f"Fraction of interactions in testing: {len(test_data) / (len(train_data) + len(test_data)):.2}\n"
        f"-------------------------------------------------\n"
        f"Number of proteins in training: {len(train_prots)} ({len(train_prots) / len(prots):.2})\n"
        f"Number of proteins in testing: {len(test_prots)} ({len(test_prots) / len(prots):.2})\n"
        f"Number of drugs in training: {len(train_drugs)} ({len(train_drugs) / len(drugs):.2})\n"
        f"Number of drugs in testing: {len(test_drugs)} ({len(test_drugs) / len(drugs):.2})\n"
        f"================================================="
    )


class Split:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset[:6] == "random":
            _, nodes, edge_num = dataset.split("x")
            self.dataset = "./" + dataset.replace("x", "_") + ".tsv"
            if os.path.exists(self.dataset):
                return
            with open(self.dataset, "w") as data:
                pd_split = int(np.random.normal(loc=0.5) * int(nodes)) + int(nodes) // 2
                drugs = [f"D{i:05d}" for i in range(pd_split)]
                prots = [f"T{i:05d}" for i in range(int(nodes) - pd_split)]
                data.write("Drug_ID\tTarget_ID\tY\n")

                edges = set()
                for i in range(int(edge_num)):
                    drug = np.random.choice(drugs)
                    prot = np.random.choice(prots)
                    while (drug, prot) in edges:
                        drug = np.random.choice(drugs)
                        prot = np.random.choice(prots)

                    edges.add((drug, prot))
                    data.write("\t".join([drug, prot, str(np.random.normal())]) + "\n")


class MinCutSplit(Split):
    def __init__(self, dataset):
        super(MinCutSplit, self).__init__(dataset)
        self.graph = nx.Graph()
        with open(self.dataset, "r") as data:
            for line in data.readlines()[1:]:
                prot, drug = line.split("\t")[:2]
                self.graph.add_node(prot)
                self.graph.add_node(drug)
                self.graph.add_edge(prot, drug, capacity=1)

    def split(self, train_partition):
        self.__partition(train_partition)

    @staticmethod
    def __karger(graph):
        start_edge = np.random.choice(graph.nodes, size=2, replace=False)
        return nx.minimum_cut(graph, start_edge[0], start_edge[1])

    def __partition(self, training):
        g = self.graph.copy()
        if training < 0.5:
            training = 1 - training

        score, new_score = len(g), len(g)
        train, test = set(), set()
        best = train, test
        while True:
            _, (p1, p2) = self.__karger(g)
            max_p = max(p1, p2, key=len)
            min_p = min(p1, p2, key=len)
            train = train.union(max_p).difference(min_p)
            test = test.union(min_p).difference(max_p)

            new_score = (len(train) / len(test)) - (training / (1 - training))
            if new_score > 0:
                g = g.subgraph(train)
            else:
                g = g.subgraph(test)

            new_score = abs(1 - new_score)
            print(new_score, "|", len(train), "|", len(test))
            if new_score < score:
                best = copy(train), copy(test)
                score = new_score
            else:
                break

        return best


class GeneticSplit(Split):
    def __init__(self, dataset):
        super(GeneticSplit, self).__init__(dataset)
        self.data = pd.read_csv(self.dataset, sep="\t")
        self.prots = self.data["Target_ID"].unique()
        self.drugs = self.data["Drug_ID"].unique()
        self.train_partition = None

    def split(self, train_partition=0.7):
        """Initialize the genetic algorithm based on some hyperparameter"""
        self.train_partition = train_partition
        ga = GA(
            num_generations=params["GA"]["G"],
            num_parents_mating=params["GA"]["P"],
            fitness_func=lambda x, y: self.fitness_function(x, y),
            sol_per_pop=10,
            num_genes=len(self.prots) + len(self.drugs),
            gene_type=int,
            keep_parents=params["GA"]["KP"],
            crossover_probability=params["GA"]["CP"],
            mutation_probability=params["GA"]["MP"],
            mutation_by_replacement=True,
            gene_space=[0, 1],
            on_generation=lambda x: self.generation_end(x),
        )
        ga.run()
        solution, solution_fitness, solution_idx = ga.best_solution()
        show_data_split(solution, self.dataset)
        return solution, solution_fitness

    def fitness_function(self, solution, idx):
        """Evaluate the intermediate solution"""
        # split the data as suggested by the solution array
        train_solution = [bool(x) for x in solution]
        test_solution = [not bool(x) for x in solution]
        train_prots = self.prots[train_solution[:len(self.prots)]]
        test_prots = self.prots[test_solution[:len(self.prots)]]
        train_drugs = self.drugs[train_solution[len(self.prots):]]
        test_drugs = self.drugs[test_solution[len(self.prots):]]

        # check if any group of [train|test] [proteins|drugs] is empty -> Return minus infinity
        if any([len(x) == 0 for x in [train_prots, test_prots, train_drugs, test_drugs]]):
            return float("-inf")

        # extract the train, test, and dropped interactions
        drop_data = self.data[
            (self.data["Target_ID"].isin(train_prots) & self.data["Drug_ID"].isin(test_drugs)) |
            (self.data["Target_ID"].isin(test_prots) & self.data["Drug_ID"].isin(train_drugs))
            ]
        train_data = self.data[self.data["Target_ID"].isin(train_prots) & self.data["Drug_ID"].isin(train_drugs)]
        test_data = self.data[self.data["Target_ID"].isin(test_prots) & self.data["Drug_ID"].isin(test_drugs)]

        """
        actually compute the score to minimize the number of dropped interactions as well as the differences between 
        the rations of drugs, targets, and interactions between train set and test set.
        As the genetic algorithm is a maximization algorithm, we have to negate the minimization score
        """
        return - (
                params["GA"]["D"] * len(drop_data) +
                params["GA"]["B"] * (
                        (1 - ((len(train_data) / len(test_data)) -
                              self.train_partition / (1 - self.train_partition))) ** 2 +
                        (1 - ((len(train_drugs) / len(test_drugs)) -
                              self.train_partition / (1 - self.train_partition))) ** 2 +
                        (1 - ((len(train_prots) / len(test_prots)) -
                              self.train_partition / (1 - self.train_partition))) ** 2
                )
        )

    @staticmethod
    def generation_end(ga_instance):
        """Track the process of optimization"""
        tmp = ga_instance.last_generation_fitness
        if ga_instance.generations_completed % 10 == 0:
            print(f"{ga_instance.generations_completed:5} | "
                  f"{min(tmp):7.4} | "
                  f"{np.mean([x for x in ga_instance.last_generation_fitness if x != float('-inf')]):7.4} | "
                  f"{max(tmp):7.4}")


methods = {
    "mincut": MinCutSplit,
    "ga": GeneticSplit,
}

if __name__ == '__main__':
    # double_cold_split_poc.py random_100_1000 ga 0.7
    methods[sys.argv[2]](sys.argv[1]).split(float(sys.argv[3]))
