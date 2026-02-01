from __future__ import annotations

import itertools
from collections import defaultdict
from functools import partial, update_wrapper
from typing import Callable, Dict, Hashable, Iterable, KeysView, List, Mapping, Optional, Set, Tuple, Union
from warnings import warn

import pygambit

from pycid.core.cpd import Outcome, StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase

# Type alias for infoset key: (agent, ((parent, value), ...))
InfosetKey = Tuple[Hashable, Tuple[Tuple[str, Outcome], ...]]

# Union of behavior and strategy profiles returned by solvers
ProfileType = Union[pygambit.MixedBehaviorProfile, pygambit.MixedStrategyProfile]


def macid_to_efg(
    macid: MACIDBase,
    decisions_in_sg: Optional[Union[KeysView[str], Set[str]]] = None,
    agents_in_sg: Optional[Iterable[Hashable]] = None,
) -> Tuple[pygambit.Game, Mapping[InfosetKey, pygambit.Infoset]]:
    """
    Creates a pygambit EFG from a MACID:
    1) Finds the MACID nodes needed for the EFG (decision nodes and informational parents S = {D u Pa_D})
    2) Creats an ordering of these nodes such that X_j precedes X_i in the ordering if and only if X_j
    is a descendant of X_i in the MACID.
    3) Labels each node X_i with a partial instantiation of the splits in the path to X_i in the EFG.
    Args:
    - macid: The MACID object to convert to a pygambit EFG.
      Expected input is a MACID object, but also allows for CIDs to be converted to EFGs.
    - decisions_in_sg: The decisions to include in the EFG. If None, all decisions are included.
    - agents_in_sg: The agents to include in the EFG. If None, all agents are included.
    Returns:
    - game: The pygambit game object. This can be used for further manipulation of the game within pygambit.
    - parents_to_infoset: A mapping from (agent, (parent, instantiation), ...) to pygambit infoset.
    """

    # can use on a subgame if copying, else do the whole game
    if decisions_in_sg is None:
        decisions_in_sg = macid.decisions
    if agents_in_sg is None:
        agents_in_sg = macid.agents

    # choose only relevant nodes
    game_tree_nodes = set(
        list(decisions_in_sg) + [parent for dec in decisions_in_sg for parent in macid.get_parents(dec)]
    )
    # topologically order them
    sorted_game_tree_nodes = macid.get_valid_order(game_tree_nodes)
    # create the pygambit efg
    game = pygambit.Game.new_tree()

    agent_to_player = _add_players(game, agents_in_sg)

    # key is instantiation of parents, value is pygambit infoset
    parents_to_infoset: Dict[InfosetKey, pygambit.Infoset] = defaultdict(dict)
    # nodes referenced in the game tree. Root has node_idx (0,), rest are (0, n, m, ...)
    # state is a dict of node_idx:state of partial instantiations of nodes
    node_idx_to_state: Dict[Tuple[int, ...], Dict[str, Outcome]] = defaultdict(dict)
    # get cardinality of each node
    num_children = [1] + [len(macid.model.domain[node]) for node in sorted_game_tree_nodes]
    range_num_children = [list(range(x)) for x in num_children]

    # grow the tree in topological order, breadth first
    for i, node in enumerate(sorted_game_tree_nodes, start=1):
        # iterate over all possible parents of the node
        for node_idx in itertools.product(*range_num_children[:i]):
            # get current node
            cur_node = _get_cur_node(game, node_idx)
            parents = macid.get_parents(node)
            parents_actions = {parent: node_idx_to_state[node_idx][parent] for parent in parents}

            # if the node is a decision, consider infosets
            if node in decisions_in_sg:
                # get agent and domain
                agent = macid.decision_agent[node]
                player = agent_to_player[agent]
                actions = macid.model.domain[node]
                action_labels = [str(a) for a in actions]
                parents_actions_tuple: InfosetKey = (agent, tuple(parents_actions.items()))
                # check if this matches an existing infoset
                if parents_actions_tuple in parents_to_infoset:
                    cur_infoset = parents_to_infoset[parents_actions_tuple]
                    game.append_infoset(cur_node, cur_infoset)
                # else create a new infoset
                else:
                    game.append_move(cur_node, player, action_labels)
                    cur_infoset = cur_node.infoset
                    # label with the node for easy reference
                    cur_infoset.label = node
                    # add to infosets
                    parents_to_infoset[parents_actions_tuple] = cur_infoset
                # add state info
                for action_idx, action in enumerate(actions):
                    state_info = node_idx_to_state[node_idx].copy()
                    state_info.update({node: action})
                    node_idx_to_state[node_idx + (action_idx,)] = state_info
            else:
                # otherwise is a chance node
                factor = macid.query([node], context=parents_actions)
                actions = macid.model.domain[node]
                action_labels = [str(a) for a in actions]
                # Create the chance move
                game.append_move(cur_node, game.players.chance, action_labels)
                chance_infoset = cur_node.infoset
                # Set all chance probabilities at once
                probs = [float(p) for p in factor.values]
                game.set_chance_probs(chance_infoset, probs)
                # add state info
                for action_idx, action in enumerate(actions):
                    state_info = node_idx_to_state[node_idx].copy()
                    state_info.update({node: action})
                    node_idx_to_state[node_idx + (action_idx,)] = state_info

    game = _add_payoffs(macid, game, range_num_children, node_idx_to_state, agents_in_sg)

    return game, parents_to_infoset


def macid_to_gambit_file(macid: MACIDBase, filename: str = "macid.efg") -> bool:
    """Converts MACID to a ".efg" file for use with GAMBIT GUI.
    Args:
    - macid: The MACID object to convert to a pygambit EFG.
    - filename: The filename to save the EFG to (default: macid.efg)."""
    game, _ = macid_to_efg(macid)
    with open(filename, "w") as f:
        f.write(game.to_efg())
    print("\nGambit .efg file has been created from the macid")

    return True


def pygambit_ne_solver(
    game: pygambit.Game, solver_override: Optional[str] = None
) -> List[ProfileType]:
    """Uses pygambit to find the Nash equilibria of the EFG.
    Default solver is enummixed for 2 player games. This finds all NEs.
    For non-2-player games, the default is enumpure which finds all pure NEs.
    If no pure NEs are found, then simpdiv is used to find a mixed NE if it exists.
    If a specific solver is desired, it can be passed as a string, but if it is not compatible
    with the game, a warning will be raised and it will be ignored. We need to do this because
    enummixed is not compatible for non-2-player games.
    Returns a list of behaviour strategies corresponding to NEs.
    """
    # check if a 2 player game, if so, default to enummixed, else enumpure
    two_player = len(game.players) == 2
    if solver_override is None:
        solver = "enummixed" if two_player else "enumpure"
    elif solver_override in ["enummixed", "lcp", "lp"] and not two_player:
        warn(f"Solver {solver_override} not allowed for non-2 player games. Using 'enumpure' instead.")
        solver = "enumpure"
    else:
        solver = solver_override

    if solver == "enummixed":
        result = pygambit.nash.enummixed_solve(game, rational=False)
    elif solver == "enumpure":
        result = pygambit.nash.enumpure_solve(game)
        # if no pure NEs found, try simpdiv if not overridden by user
        if len(result.equilibria) == 0 and solver_override is None:
            warn("No pure NEs found using enumpure. Trying simpdiv.")
            start_profile = game.mixed_strategy_profile(rational=True)
            result = pygambit.nash.simpdiv_solve(start_profile)
    elif solver == "lcp":
        result = pygambit.nash.lcp_solve(game, rational=False)
    elif solver == "lp":
        result = pygambit.nash.lp_solve(game, rational=False)
    elif solver == "simpdiv":
        # simpdiv requires a starting profile, not a game
        start_profile = game.mixed_strategy_profile(rational=True)
        result = pygambit.nash.simpdiv_solve(start_profile)
    elif solver == "ipa":
        result = pygambit.nash.ipa_solve(game)
    elif solver == "gnm":
        result = pygambit.nash.gnm_solve(game)
    else:
        raise ValueError(f"Solver {solver} not recognised")

    # Extract equilibria from NashComputationResult
    mixed_strategies = result.equilibria

    # convert to behavior strategies (except lp/lcp which return strategy profiles directly)
    behavior_strategies: List[ProfileType] = [
        x.as_behavior() if solver not in ["lp", "lcp"] else x for x in mixed_strategies
    ]

    return behavior_strategies


def behavior_to_cpd(
    macid: MACIDBase,
    parents_to_infoset: Mapping[InfosetKey, pygambit.Infoset],
    behavior: pygambit.MixedBehaviorProfile,
    decisions_in_sg: Optional[Union[KeysView[str], Set[str]]] = None,
) -> List[StochasticFunctionCPD]:
    """Convert a pygambit behavior strategy to list of CPDs for each decision node.
    Args:
    - macid: The MACID object relating to the behavior strategy.
    - parents_to_infoset: A mapping from (agent, (parent, instantiation), ...) to pygambit infoset.
    - behavior: The pygambit behavior strategy.
    - decisions_in_sg: The decisions in the subgame. If None, use all decisions in the MACID.
    Returns:
    - cpds: A list of CPDs for each decision node.
    """

    def _action_prob_given_parents(node: str, **pv: Outcome) -> Mapping[Outcome, float]:
        """Takes the parent instantiation and outputs the prob from the infoset"""
        pv_tuple: InfosetKey = (macid.decision_agent[node], tuple(pv.items()))
        # get the infoset for the node
        infoset = parents_to_infoset[pv_tuple]
        # if the infoset does not exist, this is not a valid parent instantiation
        if not infoset:
            return {}
        # get the action probs for the infoset
        # In pygambit 16.5.0, iterating behavior[infoset] yields (Action, probability) tuples
        action_probs = {
            macid.model.domain[node][i]: float(action_prob[1])
            for i, action_prob in enumerate(behavior[infoset])
        }
        return action_probs

    def _wrapped_partial(
        func: Callable[..., Mapping[Outcome, float]], *args: str
    ) -> Callable[..., Mapping[Outcome, float]]:
        """Adds __name__ and __doc__ to partial functions"""
        partial_func = partial(func, *args)
        update_wrapper(partial_func, func)
        return partial_func

    if decisions_in_sg is None:
        decisions_in_sg = macid.decisions

    # require domain to get cpd.values in the same order as in macid
    cpds = [
        StochasticFunctionCPD(
            variable=node,
            stochastic_function=_wrapped_partial(_action_prob_given_parents, node),
            cbn=macid,
            domain=macid.model.domain[node],
        )
        for node in decisions_in_sg
    ]
    return cpds


def _add_players(game: pygambit.Game, agents_in_sg: Iterable[Hashable]) -> Dict[Hashable, pygambit.Player]:
    """add players to the pygambit game"""
    agent_to_player: Dict[Hashable, pygambit.Player] = {}
    for agent in agents_in_sg:
        player = game.add_player(str(agent))
        agent_to_player[agent] = player

    return agent_to_player


def _add_payoffs(
    macid: MACIDBase,
    game: pygambit.Game,
    range_num_children: List[List[int]],
    node_idx_to_state: Dict[Tuple[int, ...], Dict[str, Outcome]],
    agents_in_sg: Iterable[Hashable],
) -> pygambit.Game:
    """add payoffs to the game as leave nodes"""
    agents_list = list(agents_in_sg)
    for node_idx in itertools.product(*range_num_children):
        cur_node = _get_cur_node(game, node_idx)
        context = node_idx_to_state[node_idx]
        # compute payoffs for all agents
        payoffs = [float(macid.expected_utility(context=context, agent=agent)) for agent in agents_list]
        # create outcome with payoffs and label
        outcome = game.add_outcome(payoffs, label=str(node_idx))
        game.set_outcome(cur_node, outcome)

    return game


def _get_cur_node(game: pygambit.Game, idx: Tuple[int, ...]) -> pygambit.Node:
    """Returns the current node in the game tree given the index of the node."""
    cur_node = game.root
    # first entry is the root node
    if len(idx) == 1:
        return cur_node
    # traverse the tree
    for i in idx[1:]:
        cur_node = cur_node.children[i]
    return cur_node
