"""Evolutionary clustering for jointly evaluating decision and objective spaces."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score


def evolutionary_multi_space_clustering(
    decision_space,
    objective_space,
    n_clusters,
    population_size=30,
    generations=40,
    mutation_rate=0.03,
    immigrant_fraction=0.1,
    elite_size=2,
    tournament_size=3,
    sample_size=1000,
    patience=12,
    random_state=42,
):
    """Return fixed-k labels maximizing mean silhouette across two spaces.

    Parents are selected by tournament, retained elites are copied unchanged,
    and offspring use label-aligned uniform crossover with cluster repair.
    Mutation preserves nonempty clusters; random immigrants maintain diversity.
    """
    decision_space = np.asarray(decision_space, dtype=float)
    objective_space = np.asarray(objective_space, dtype=float)
    if decision_space.ndim != 2 or objective_space.ndim != 2:
        raise ValueError("Both spaces must be two-dimensional arrays.")
    if decision_space.shape[0] != objective_space.shape[0]:
        raise ValueError("Both spaces must contain the same observations in the same order.")

    n_samples = decision_space.shape[0]
    if decision_space.shape[1] == 0 or objective_space.shape[1] == 0:
        raise ValueError("Both spaces must contain at least one feature.")
    if not 2 <= n_clusters < n_samples:
        raise ValueError("n_clusters must be at least 2 and smaller than the number of rows.")
    if population_size < 2 or not 1 <= elite_size < population_size:
        raise ValueError("Use population_size >= 2 and 1 <= elite_size < population_size.")
    if generations < 1 or patience < 1 or tournament_size < 1:
        raise ValueError("generations, patience, and tournament_size must be positive.")
    if not 0 <= mutation_rate <= 1 or not 0 <= immigrant_fraction < 1:
        raise ValueError("mutation_rate must be in [0, 1] and immigrant_fraction in [0, 1).")
    if sample_size is not None and sample_size < 2:
        raise ValueError("sample_size must be at least 2 or None for exact scoring.")
    if not np.isfinite(decision_space).all() or not np.isfinite(objective_space).all():
        raise ValueError("Input spaces must contain only finite values.")

    rng = np.random.default_rng(random_state)
    score_cache = {}

    def canonicalize(labels):
        _, first_indices, inverse = np.unique(labels, return_index=True, return_inverse=True)
        ordered_labels = np.argsort(first_indices)
        remapping = np.empty_like(ordered_labels)
        remapping[ordered_labels] = np.arange(len(ordered_labels))
        return remapping[inverse]

    def score(labels):
        labels = canonicalize(labels)
        key = tuple(labels)
        if key not in score_cache:
            sample_count = min(sample_size, n_samples) if sample_size is not None else None
            decision_score = silhouette_score(
                decision_space,
                labels,
                sample_size=sample_count,
                random_state=random_state,
            )
            objective_score = silhouette_score(
                objective_space,
                labels,
                sample_size=sample_count,
                random_state=random_state,
            )
            score_cache[key] = 0.5 * (decision_score + objective_score)
        return score_cache[key]

    def random_partition():
        labels = np.arange(n_samples) % n_clusters
        rng.shuffle(labels)
        return labels

    def select(population, scores):
        contenders = rng.choice(
            len(population),
            size=min(tournament_size, len(population)),
            replace=False,
        )
        return population[contenders[np.argmax(scores[contenders])]]

    def crossover(parent_a, parent_b):
        overlap = np.zeros((n_clusters, n_clusters), dtype=int)
        np.add.at(overlap, (parent_a, parent_b), 1)
        rows, columns = linear_sum_assignment(-overlap)
        label_mapping = np.empty(n_clusters, dtype=int)
        label_mapping[columns] = rows
        aligned_b = label_mapping[parent_b]
        child = np.where(rng.random(n_samples) < 0.5, parent_a, aligned_b)

        counts = np.bincount(child, minlength=n_clusters)
        for missing_label in np.flatnonzero(counts == 0):
            eligible = np.flatnonzero(counts[child] > 1)
            supported = eligible[
                (parent_a[eligible] == missing_label)
                | (aligned_b[eligible] == missing_label)
            ]
            candidates = supported if len(supported) else eligible
            point = rng.choice(candidates)
            counts[child[point]] -= 1
            child[point] = missing_label
            counts[missing_label] += 1
        return child

    def mutate(labels):
        labels = labels.copy()
        counts = np.bincount(labels, minlength=n_clusters)
        n_mutations = int(round(mutation_rate * n_samples))
        if n_mutations == 0 and mutation_rate > 0:
            n_mutations = 1
        for point in rng.choice(n_samples, size=n_mutations, replace=False):
            current_label = labels[point]
            if counts[current_label] <= 1:
                continue
            other_labels = np.delete(np.arange(n_clusters), current_label)
            new_label = rng.choice(other_labels)
            labels[point] = new_label
            counts[current_label] -= 1
            counts[new_label] += 1
        return labels

    population = [random_partition() for _ in range(population_size)]
    best_labels = None
    best_score = -np.inf
    stale_generations = 0

    for _ in range(generations):
        scores = np.asarray([score(labels) for labels in population])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_labels = canonicalize(population[order[0]]).copy()
            stale_generations = 0
        else:
            stale_generations += 1
        if stale_generations >= patience:
            break

        next_population = [population[index].copy() for index in order[:elite_size]]
        immigrant_count = min(
            population_size - elite_size,
            int(round(population_size * immigrant_fraction)),
        )
        offspring_count = population_size - elite_size - immigrant_count
        while len(next_population) < elite_size + offspring_count:
            parent_a = select(population, scores)
            parent_b = select(population, scores)
            next_population.append(mutate(crossover(parent_a, parent_b)))
        next_population.extend(random_partition() for _ in range(immigrant_count))
        population = next_population

    return best_labels