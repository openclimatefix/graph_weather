"""Models"""

# Using lazy loading to avoid dependency conflicts

__all__ = []

def __getattr__(name):
    """Lazy loading for models to avoid dependency conflicts."""
    if name in ['ImageMetaModel', 'LoRAModule', 'MetaModel', 'WrapperImageModel', 'WrapperMetaModel']:
        from .fengwu_ghr.layers import (
            ImageMetaModel as IM,
            LoRAModule as LM,
            MetaModel as MM,
            WrapperImageModel as WIM,
            WrapperMetaModel as WMM,
        )
        result = {
            'ImageMetaModel': IM,
            'LoRAModule': LM,
            'MetaModel': MM,
            'WrapperImageModel': WIM,
            'WrapperMetaModel': WMM,
        }[name]
        globals()[name] = result
        return result
    elif name == 'AssimilatorDecoder':
        from .layers.assimilator_decoder import AssimilatorDecoder as AD
        globals()[name] = AD
        return AD
    elif name == 'AssimilatorEncoder':
        from .layers.assimilator_encoder import AssimilatorEncoder as AE
        globals()[name] = AE
        return AE
    elif name == 'Decoder':
        from .layers.decoder import Decoder as D
        globals()[name] = D
        return D
    elif name == 'Encoder':
        from .layers.encoder import Encoder as E
        globals()[name] = E
        return E
    elif name == 'Processor':
        from .layers.processor import Processor as P
        globals()[name] = P
        return P
    elif name == 'StochasticDecompositionLayer':
        from .layers.stochastic_decomposition import StochasticDecompositionLayer as SDL
        globals()[name] = SDL
        return SDL
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
