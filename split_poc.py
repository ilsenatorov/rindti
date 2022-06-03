import os.path
import sys
import networkx as nx
import numpy as np
from copy import copy
import pandas as pd
from pygad import GA
from pytorch_lightning import seed_everything
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


params = {
    "GA": {
        "G": 1000,  # number of generations in optimization
        "P": 3,  # number of parents
        "KP": 2,  # keep parents to next generation
        "MP": 0.1,  # mutation probability
        "CP": 0.2,  # crossover probability
        "D": 1,  # weight of drug balance between train and test sets
        "B": 1,  # weight of protein balance between train and test sets
    }
}


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def show_data_split(split, dataset, train_partition, output=True):
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
    prob_lower_bound = 2 * (len(data) / (len(prots) + len(drugs))) * train_partition * \
        (1 - train_partition) * (len(prots) + len(drugs))

    score = - (
            params["GA"]["D"] * ((len(data) - len(train_data) - len(test_data)) / prob_lower_bound) +
            params["GA"]["B"] * (
                    abs(len(train_data) / (len(data) - (len(data) - len(train_data) - len(test_data))) - train_partition) +
                    abs(len(train_drugs) / len(drugs) - train_partition) +
                    abs(len(train_prots) / len(prots) - train_partition)
            )
    )
    if output:
        print(
            f"=================================================\n"
            f"Final Split Evaluation\n"
            f"-------------------------------------------------\n"
            f"Final scoring for the solution: {score:.5}\n"
            f"Total number of interactions: {len(data)}\n"
            f"Dropped number of interactions: {len(data) - len(train_data) - len(test_data)} "
            f"({(len(data) - len(train_data) - len(test_data)) / len(data):.2})\n"
            f"Probabilistic lower bound: {int(prob_lower_bound)}\n"
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
    return score


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
        self.prob_lower_bound = None
        self.train_partition = None

    def split(self, train_partition=0.7):
        """Initialize the genetic algorithm based on some hyperparameter"""
        self.train_partition = train_partition
        self.prob_lower_bound = 2 * (len(self.data) / (len(self.prots) + len(self.drugs))) * train_partition * \
                                (1 - train_partition) * (len(self.prots) + len(self.drugs))
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
        # show_data_split(solution, self.dataset, self.train_partition)
        return solution

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

        """
        actually compute the score to minimize the number of dropped interactions as well as the differences between 
        the rations of drugs, targets, and interactions between train set and test set.
        As the genetic algorithm is a maximization algorithm, we have to negate the minimization score
        """
        return - (
                params["GA"]["D"] * (len(drop_data) / self.prob_lower_bound) +
                params["GA"]["B"] * (
                        abs(len(train_data) / (len(self.data) - len(drop_data)) - self.train_partition) +
                        abs(len(train_drugs) / len(self.drugs) - self.train_partition) +
                        abs(len(train_prots) / len(self.prots) - self.train_partition)
                )
        )

    @staticmethod
    def generation_end(ga_instance):
        """Track the process of optimization"""
        tmp = ga_instance.last_generation_fitness
        if ga_instance.generations_completed % 10000 == 0:
            print(f"{ga_instance.generations_completed:5} | "
                  f"{min(tmp):7.4} | "
                  f"{np.mean([x for x in ga_instance.last_generation_fitness if x != float('-inf')]):7.4} | "
                  f"{max(tmp):7.4}")


methods = {
    "mincut": MinCutSplit,
    "ga": GeneticSplit,
}


def eval_split(args):
    np.random.seed(42)
    results = []
    ds = None
    for seed in np.random.uniform(0, 100, int(args[3])):
        seed_everything(seed)
        splitter = methods[args[1]](args[0])
        if ds is None:
            ds = splitter.dataset
        solution = splitter.split(float(sys.argv[3]))
        results.append((solution, show_data_split(solution, ds, float(sys.argv[3]), output=False)))
    best = max(results, key=lambda x: x[1])
    show_data_split(best[0], ds, float(sys.argv[3]))


def get_scaffolds(lig):
    """Compute the scaffold for proteins"""
    if lig != lig:
        return np.nan
    mol = Chem.MolFromSmiles(lig)
    if mol is None:
        return np.nan
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic_sf = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.CanonSmiles(Chem.MolToSmiles(generic_sf))


def solve_int_knapsack(weights, border):
    """Solve the knapsack problem using dynamic programming"""
    k = [[0 for _ in range(border + 1)] for _ in range(len(weights) + 1)]
    s = [[[] for _ in range(border + 1)] for _ in range(len(weights) + 1)]
    for i in range(len(weights) + 1):
        for w in range(border + 1):
            if i == 0 or w == 0:
                k[i][w] = 0
            elif weights[i - 1] <= w:
                if 1 + k[i - 1][w - weights[i - 1]] > k[i - 1][w]:
                    k[i][w] = 1 + k[i - 1][w - weights[i - 1]]
                    s[i][w] = s[i - 1][w - weights[i - 1]] + [i - 1]
                else:
                    k[i][w] = k[i - 1][w]
                    s[i][w] = s[i - 1][w]
            else:
                k[i][w] = k[i - 1][w]
                s[i][w] = s[i - 1][w]
    return k[len(weights)][border], s[len(weights)][border]


def scaffold_split(args):
    """Compute a scaffold split for drugs"""
    # read in the data and compute scaffolds
    inter = pd.read_csv(os.path.join(args[0], "tables", "inter.tsv"), sep="\t")
    ligs = pd.read_csv(os.path.join(args[0], "tables", "lig.tsv"), sep="\t")
    # ligs = ligs.dropna(subset=["Scaff"])
    ligs["Scaff"] = ligs["Drug"].apply(lambda x: get_scaffolds(x))

    # find the groups for each scaffold and compute it's size in interactions in the actual data
    groups = {}
    for i, (_, value) in enumerate(ligs.groupby("Scaff").indices.items()):
        count = 0
        drugs = []
        for index in value:
            drugs.append(ligs.loc[index, "Drug_ID"])
            count += len(inter[inter["Drug_ID"] == drugs[-1]])
        groups[i] = (drugs, count)

    # due to better runtime, we find the data that go into test data,
    # the knapsack solver's runtime depends on the border
    upper_limit = int(len(inter) * (1 - float(args[1])))
    test_indices = solve_int_knapsack(
        [groups[x][1] for x in groups.keys()],
        upper_limit,
    )

    # assign the actual separation into train and test split
    test_drugs = flatten_list([groups[i][0] for i in test_indices[1]])
    inter["split"] = inter["Drug_ID"].apply(lambda x: "test" if x in test_drugs else "train")
    inter.to_csv(os.path.join(args[0], "tables", "inter_scaff.tsv"), sep="\t", index=False)


if __name__ == '__main__':
    """
    For double cold splitting:
    splitting_poc.py dc random_100_1000 ga 0.7 5 (script, split-mode, graph, training-split, runs of algorithm)
    
    For scaffold split:
    splitting_poc.py sc /path/to/resources 0.7 (script, split-mode, data, training-split)
    """
    if sys.argv[1] == "dc":
        eval_split(sys.argv[2:])
    elif sys.argv[1] == "sc":
        scaffold_split(sys.argv[2:])
