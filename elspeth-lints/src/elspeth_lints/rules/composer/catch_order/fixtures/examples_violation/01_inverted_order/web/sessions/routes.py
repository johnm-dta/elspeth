class ComposerServiceError(Exception):
    pass


class ComposerPluginCrashError(ComposerServiceError):
    pass


def f():
    try:
        pass
    except ComposerServiceError:
        pass
    except ComposerPluginCrashError:
        pass
