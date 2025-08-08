from collections.abc import Mapping, Sequence
from flax import nnx
from jaxlip.linear import SpectralLinear, OrthoLinear
from jaxlip.batchop import BatchCentering2d, BatchCentering, LayerCentering
from jaxlip.conv import SpectralConv2d, AOLConv2d


def cache_model_params(root: nnx.Module, verbose: bool = True) -> None:
    """
    Recursively traverse `root` and invoke `specific_method()` on
    every `nnx.Linear` instance in the tree.

    Parameters
    ----------
    root : nnx.Module
        The top-level Flax nnx model (or sub-module) to traverse.

    Notes
    -----
    * A small `visited` set prevents re-visiting the same object when
      modules are shared.
    * The traversal looks inside standard Python containers
      (dict / list / tuple / set) so nested collections work too.
    """
    visited: set[int] = set()

    def _walk(mod: nnx.Module) -> None:
        if mod == root and verbose:
            print(
                "WARNING: Do not retrain this network without previously running ._uncache()"
            )
        if id(mod) in visited:  # avoid duplicates
            return
        visited.add(id(mod))

        if isinstance(mod, (OrthoLinear, SpectralLinear, SpectralConv2d, AOLConv2d)):
            mod._cache_params()
        if isinstance(
            mod,
            (
                BatchCentering2d,
                BatchCentering,
            ),
        ):
            mod.use_running_average = True

        for _, value in vars(mod).items():
            if isinstance(value, nnx.Module):
                _walk(value)
            elif isinstance(value, Mapping):
                for v in value.values():
                    if isinstance(v, nnx.Module):
                        _walk(v)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for v in value:
                    if isinstance(v, nnx.Module):
                        _walk(v)

    _walk(root)
    pass


def uncache_model_params(root: nnx.Module) -> None:
    """
    Recursively traverse `root` and invoke `specific_method()` on
    every `nnx.Linear` instance in the tree.

    Parameters
    ----------
    root : nnx.Module
        The top-level Flax nnx model (or sub-module) to traverse.

    Notes
    -----
    * A small `visited` set prevents re-visiting the same object when
      modules are shared.
    * The traversal looks inside standard Python containers
      (dict / list / tuple / set) so nested collections work too.
    """
    visited: set[int] = set()

    def _walk(mod: nnx.Module) -> None:
        if id(mod) in visited:  # avoid duplicates
            return
        visited.add(id(mod))

        if isinstance(mod, (OrthoLinear, SpectralLinear, SpectralConv2d, AOLConv2d)):
            mod._uncache()
        if isinstance(
            mod,
            (
                BatchCentering2d,
                BatchCentering,
            ),
        ):
            mod.use_running_average = False

        for _, value in vars(mod).items():
            if isinstance(value, nnx.Module):
                _walk(value)
            elif isinstance(value, Mapping):
                for v in value.values():
                    if isinstance(v, nnx.Module):
                        _walk(v)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for v in value:
                    if isinstance(v, nnx.Module):
                        _walk(v)

    _walk(root)
    pass
