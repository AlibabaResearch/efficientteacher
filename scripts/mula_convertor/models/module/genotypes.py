from copy import deepcopy

class YoloStructure:
    def __init__(self,nas_channels,metric=0.,flops=0.,params=0.):
        self.nas_channels = nas_channels
        self.metric = metric
        self.flops = flops
        self.params = params
    def mutate(self,mutate_ratio):
        pass


def regularized_evolution(
    cycles,
    population_size,
    sample_size,
    time_budget,
    random_arch,
    mutate_arch,
    nas_bench,
    extra_info,
    dataname,
):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each tournament.
      time_budget: the upper bound of searching cost

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history, total_time_cost = (
        [],
        0,
    )  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_arch()
        model.accuracy, time_cost = train_and_eval(
            model.arch, nas_bench, extra_info, dataname
        )
        population.append(model)
        history.append(model)
        total_time_cost += time_cost

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    # while len(history) < cycles:
    while total_time_cost < time_budget:
        # Sample randomly chosen models from the current population.
        start_time, sample = time.time(), []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        total_time_cost += time.time() - start_time
        child.accuracy, time_cost = train_and_eval(
            child.arch, nas_bench, extra_info, dataname
        )
        if total_time_cost + time_cost > time_budget:  # return
            return history, total_time_cost
        else:
            total_time_cost += time_cost
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
    return history, total_time_cost