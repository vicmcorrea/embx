class EmbxError(Exception):
    pass


class ValidationError(EmbxError):
    pass


class ConfigurationError(EmbxError):
    pass


class ProviderError(EmbxError):
    pass
