# codes/models/__init__.py
import logging

def create_model(opt):
    """
    Factory that returns an initialized model instance based on opt['model'].
    Valid values: 'sr3' (our diffusion SR with wavelet prior), 'sr' (regression SR).
    """
    model_name = str(opt.get('model', '')).lower()

    if model_name == 'sr3':
        from .SR3_model import SR3Model as M
    elif model_name == 'sr':
        from .SR_model import SRModel as M
    else:
        raise NotImplementedError(f"Model [{model_name}] not recognized. "
                                  f"Expected one of: 'sr3', 'sr'.")

    m = M(opt)
    logging.getLogger('base').info(f"Model [{m.__class__.__name__}] is created.")
    return m

__all__ = ['create_model']
