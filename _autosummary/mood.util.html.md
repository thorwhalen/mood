# mood.util

Utils for mood

### Functions

| [`add_egress`](#mood.util.add_egress)(post)   | Decorator to add an egress to the output of a function.   |
|---------------------------------------------------------------------|-----------------------------------------------------------|

### mood.util.add_egress(post)

Decorator to add an egress to the output of a function.

```pycon
>>> @add_egress(str)
... def foo(x: int) -> int:
...     return x + 1
>>> foo(3)
'4'
```

Note that if you need to access the underlying wrapped function, you can do so in
the python-standard way:

```pycon
>>> underlying_foo = foo.__wrapped__
>>> underlying_foo(3)
4
```
