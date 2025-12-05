.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/AffinityPropagation.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/AffinityPropagation
    .. image:: https://readthedocs.org/projects/AffinityPropagation/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://AffinityPropagation.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/AffinityPropagation/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/AffinityPropagation
    .. image:: https://img.shields.io/pypi/v/AffinityPropagation.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/AffinityPropagation/
    .. image:: https://img.shields.io/conda/vn/conda-forge/AffinityPropagation.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/AffinityPropagation
    .. image:: https://pepy.tech/badge/AffinityPropagation/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/AffinityPropagation
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/AffinityPropagation

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===================
AffinityPropagation
===================

    Implementation: Adrian Homa

    Testing: Maciej Kucharski
 Introduction
Affinity Propagation is a clustering algorithm that automatically finds the number of clusters in a dataset. The algorithm identifies exemplars, which are representative data points that serve as cluster centers.
 Basic Idea
The algorithm works by treating every data point as a potential cluster center. Points then exchange messages with each other to decide which points should become exemplars and which points should belong to which cluster. This message passing continues until the algorithm reaches a stable configuration.
 Algorithm Steps
1. Computing Similarities

First, we calculate how similar each pair of points is to each other. We use the negative squared Euclidean distance as our similarity measure:
S(i,j) = -||xi - xj||^2
This means that points closer together have higher similarity values. The diagonal values S(k,k) are set to a preference parameter that controls how likely each point is to become an exemplar.
2. Message Passing

The algorithm updates two types of messages between all pairs of points:
**Responsibility r(i,k)**: This message indicates how good point k could be as an exemplar for point i, compared to all other potential exemplars. It's calculated as:
r(i,k) = S(i,k) - max over k' /= k of {a(i,k') + S(i,k')}
it's the similarity between i and k, minus the strongest competition from others.
**Availability a(i,k)**: How appropriate it is for point i to choose point k as exemplar, based on support from other points. For non-diagonal elements:
a(i,k) = min(0, r(k,k) + sum of max(0, r(i',k)) for all i' not in {i,k})
For diagonal elements (self-availability):
a(k,k) = sum of max(0, r(i',k)) for all i' not equal to k
3. Damping

To prevent the algorithm from oscillating between solutions, we apply damping when updating the messages. Instead of completely replacing old values with new ones, we use a weighted average:
R_new = lambda * R_old + (1 - lambda) * R_computed
The same applies to availability. The damping factor lambda is typically set between 0.5 and 0.9.
4. Finding Exemplars

After each iteration, we identify exemplars by looking at points where r(k,k) + a(k,k) > 0. These are the points that have enough support to be cluster centers.
5. Assigning Clusters

Once we have our exemplars, each point is assigned to whichever exemplar it's most similar to. This gives us our final clustering.
6. Stopping Condition

The algorithm stops when the set of exemplars doesn't change for a certain number of iterations (15) or when we reach the maximum number of iterations.
 Parameters

**damping**: Controls how much we trust new information versus old information. Range is 0.5 to 1.0, with higher values making updates more conservative. Default is 0.5.

**max_iter**: Maximum number of iterations before stopping. Default is 200.

**convergence_iter**: How many iterations with the same exemplars before we declare convergence. Default is 15.

**preference**: Influences how many clusters we get. Higher values lead to more clusters. If set to None, the algorithm uses the median of all similarities.

**affinity**: How to compute the similarity matrix. Can be 'euclidean' for automatic computation or 'precomputed' if you provide your own similarity matrix.

 Tips for Use
**Adjusting the number of clusters**: If you're getting too many clusters, decrease the preference value. If you're getting too few, increase it.

**Convergence issues**: If the exemplars keep changing, try increasing the damping factor closer to 1.0.

**Speed**: For large datasets, consider using a subset of your data first to estimate good parameter values, then run on the full dataset.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
