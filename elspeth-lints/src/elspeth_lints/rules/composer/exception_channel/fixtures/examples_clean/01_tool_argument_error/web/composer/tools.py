class ToolArgumentError(Exception):
    pass


def f():
    raise ToolArgumentError(argument="x", expected="a string", actual_type="int")
