class ComposerServiceError(Exception):
    pass


class ComposerPluginCrashError(ComposerServiceError):
    pass


def f():
    try:
        pass
    except ComposerPluginCrashError:
        pass
    except ComposerServiceError:
        pass
