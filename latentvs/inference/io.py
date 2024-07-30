from inference.utils import VSArguments
from inference.interaction_matrix_mixer import AverageCurrentAndDesiredInteractionMatrices, CurrentInteractionMatrix, DesiredInteractionMatrix, InteractionMatrixMixer
import yaml
from inference.optimizers import *
from utils.custom_typing import *
def get_or_default(key: str, node: yaml.Node, default_node: yaml.Node) -> Any:
    return node[key] if key in node else default_node[key]

def get_or_default_val(key: str, node: yaml.Node, default_val: Any) -> Any:
    return node[key] if key in node else default_val
def optional_val(key: str, node: yaml.Node, on_present: Callable[[Any,], Any] = lambda x: x) -> Any:
    if key in node:
        return on_present(node[key])
    else:
        return None


model_builders = {}

def optimizer_from_yaml(node: yaml.Node, device: str) -> Optimizer:
    def make_lm_opt_node(node: yaml.Node) -> LevenbergMarquardtOptimizer:
        return LevenbergMarquardtOptimizer(
            node['mu_initial'],
            node['iter_gauss_newton'],
            node['mu_factor'],
            node['mu_min'],
            device
        )
    switch_type = {
        'Linear': lambda _node: LinearOptimizer(),
        'LevenbergMarquardt': make_lm_opt_node
    }
    return switch_type[node['type']](node)

def interaction_matrix_mixer_from_yaml(node: yaml.Node) -> InteractionMatrixMixer:
    return {
        'current_interaction_matrix': CurrentInteractionMatrix(),
        'desired_interaction_matrix': DesiredInteractionMatrix(),
        'ESM': AverageCurrentAndDesiredInteractionMatrices()
    }[node['type']]

def vs_method_build_fn_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
    return model_builders[node['type']](node, defaults, globals, device)

def vs_builder_from_yaml(node: yaml.Node, defaults_node: yaml.Node, globals: yaml.Node, pose_type: str, device: str) -> VSMethodBuilder:
    from aevs.inference.methods import Weighting
    def make_exp_suffix(mixer: InteractionMatrixMixer, pose_type: str, weighting: Weighting) -> str:
        suffix = '_'
        for m, mn in zip([AverageCurrentAndDesiredInteractionMatrices, CurrentInteractionMatrix, DesiredInteractionMatrix], ['ESM', 'Li', 'Lid']):
            if isinstance(mixer, m):
                suffix += mn
                break
        suffix += '_'
        suffix += pose_type
        if weighting == Weighting.DecoderError:
            suffix += '_DW'
        return suffix
    mixer = get_or_default('interaction_matrix_mixer', node, defaults_node)
    weighting = get_or_default_val('weighting', node, Weighting.Identity)
    suffix = make_exp_suffix(mixer, pose_type, weighting)
    gain = get_or_default('gain', node, defaults_node)
    name = node['name'] + suffix
    return (gain, name, vs_method_build_fn_from_yaml(node, defaults_node, globals, device))

def vs_builders_list_from_yaml(node: yaml.Node, defaults_node: yaml.Node, globals: yaml.Node, pose_type: str, device: str) -> List[VSMethodBuilder]:
    builders = []
    for run_node in node:
        vs_builder = vs_builder_from_yaml(run_node, defaults_node, globals, pose_type, device)
        builders.append(vs_builder)
    return builders
