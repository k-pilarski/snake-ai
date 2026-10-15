import snake_ai


def test_package_has_version():
    assert isinstance(snake_ai.__version__, str)
    assert snake_ai.__version__
