import pytest


def f(x):
    return x + 1


def test_f():
    assert f(1) == 2
    assert f(2) == 3
