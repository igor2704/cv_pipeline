import numpy as np
import typing as tp
from datetime import datetime
from copy import deepcopy


class UnknownDim(Exception):
    pass


class WasNotRun(Exception):
    pass


class GeneticAlgorithm:

    """
    Сreate an object that finds the maximum of the target function.
    main idea of the algorithm:
    https://www.drdobbs.com/database/differential-evolution/184410166.
    """

    def __init__(self, func: tp.Callable[..., float], mut_force: float = 0.5,
                prob_mut: float = 0.75, internal_mut_bound: float = 1e-7,
                max_iter: int = 1000, delta_converged: float = 1e-7, population_count: int = 15, dim: int | None = None,
                initial_population: np.ndarray | None = None, initial_population_bound: float | np.ndarray = 1,
                iter_increase: int = 30, increase_force: float = 0.25, decrease_increase_force: float = 0.5,
                silent: bool = False):
        """
        :param func: target function.
        :param mut_force: mutation force.
        :param prob_mut: mutation probability.
        :param internal_mut_bound: probability of internal mutation.
                Each attribute of the descendant always varies in the range from
                (-internal_mut_bound% , internal_mut_bound%) inherited.
        :param max_iter: maximum number of iterations (generations)
        :param delta_converged: if the difference between the target function
                values of the best and worst sample in population is less than delta_converged,
                then the search for the best population ends.
        :param population_count: number of samples in a population.
        :param dim: dimension of the sample vector (number of features).
        :param initial_population: initial population.
        :param initial_population_bound: if initial_population is not specified, then the
                initial population is vectors whose components are random numbers from the range
                (-initial_population_bound, initial_population_bound).
        :param iter_increase: number of iteration (population) starting from which the
                mut_force increases.
        :param increase_force: how much does mut_force increase.
        :param decrease_increase_force: how much does increase_force decrease.
        :param silent: whether intermediate information will be displayed.
        """
        self.max_iter = max_iter
        self.delta_converged = delta_converged
        self.population_count = population_count
        self.iter_increase = iter_increase
        self._history: list[tuple[np.ndarray, np.ndarray, float]] | None = None
        self.internal_mut_bound = internal_mut_bound
        self.mut_force = mut_force
        self.prob_mut = prob_mut
        self.increase_force = increase_force
        self.decrease_increase_force = decrease_increase_force
        self.func = func
        self.silent = silent
        if dim is None and initial_population is None:
            raise UnknownDim()
        elif initial_population is None and dim is not None:
            self.initial_population = (np.random.rand(population_count, dim) - 0.5) * 2 * initial_population_bound
            self.dim = dim
        elif initial_population is not None and dim is None:
            self.dim = len(initial_population)
            self.initial_population = initial_population  # type: ignore
        else:
            self.dim = dim  # type: ignore
            self.initial_population = initial_population  # type: ignore

    def run(self) -> None:
        # initialization
        population: np.ndarray = self.initial_population

        vector_func = np.vectorize(self.func)

        init_func_value: np.ndarray = vector_func(*[population[:, i] for i in range(self.dim)])
        self.best_sample: np.ndarray = population[np.argmax(init_func_value)]

        self._history = list()
        self._history.append((population, self.best_sample, init_func_value.max()))

        increase_val: float = 0
        increase_val_force: float = self.increase_force

        if not self.silent:
            print('iteration 0')
            print(f'best sample: {self.best_sample} \t' + f'best function value: {init_func_value.max()}')

        for iteration in range(self.max_iter):

            time_begin: datetime = datetime.now()

            # reproduction (creation of pairs)
            ancestors: np.ndarray = np.array([np.random.choice(self.population_count, 2, replace=False) for
                                              i in range(self.population_count)])
            anc_for_diff: np.ndarray = np.array([np.random.choice(self.population_count, 2, replace=False) for
                                                 i in range(self.population_count)])

            # birth
            genes_choice: np.ndarray = np.random.randint(2, size=self.population_count)
            genes_mutatin: np.ndarray = (np.random.random_sample(self.population_count) * 2 - 1) * \
                                                                                    self.internal_mut_bound
            children: np.ndarray = population[ancestors[:, 0]].transpose() * genes_choice.transpose() + \
                                    population[ancestors[:, 1]].transpose() * (1 - genes_choice)
            children *= (1 + genes_mutatin)
            mutation_proba: np.ndarray = np.random.choice(2, self.population_count,
                                                          p=[1 - self.prob_mut, self.prob_mut])

            if iteration % self.iter_increase == 0:
                increase_val += increase_val_force
                increase_val_force *= self.decrease_increase_force
            children += self.mut_force * (1 + increase_val) * \
                        (population[anc_for_diff[:, 0]] - population[anc_for_diff[:, 1]]).transpose() * mutation_proba
            children = children.transpose()

            # selection
            children_res: np.ndarray = vector_func(*[children[:, i] for i in range(self.dim)])
            parent_res: np.ndarray = vector_func(*[population[:, i] for i in range(self.dim)])
            population = np.array([children[i] if children_res[i] > parent_res[i] else population[i]
                                   for i in range(len(children_res))])
            population_res: np.ndarray = np.maximum(children_res, parent_res)

            self.best_sample = population[np.argmax(population_res)]
            self._history.append((population, self.best_sample, population_res.max()))

            if not self.silent:
                print(f'duration of iteration {iteration + 1}: ' + str((datetime.now() - time_begin).total_seconds()) \
                      + ' sec.')
                print(f'best sample: {self.best_sample} \t' + f'best function value: {population_res.max()}')

            if population_res.max() - parent_res.min() < self.delta_converged:
                break

    def get_history(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        :return: population history.
        """
        if self._history is None:
            raise WasNotRun()
        return deepcopy(self._history)
