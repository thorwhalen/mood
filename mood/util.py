"""
Utils for mood
"""

import os
from functools import wraps, partial

from i2 import get_app_data_folder

import dol

pkg_name = "mood"

_root_app_data_dir = get_app_data_folder()
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR",
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir, verbose=f"Making app dir: {app_data_dir}")
djoin = partial(os.path.join, app_data_dir)


# NOte: Other versions: i2.postprocess and possible with partial of i2.wrap
#   Put a simple version of this functionality here to grow deps slowly.
#   If needs to be maintained, don't: Just source it from i2.
def add_egress(post):
    """Decorator to add an egress to the output of a function.

    >>> @add_egress(str)
    ... def foo(x: int) -> int:
    ...     return x + 1
    >>> foo(3)
    '4'

    Note that if you need to access the underlying wrapped function, you can do so in
    the python-standard way:

    >>> underlying_foo = foo.__wrapped__
    >>> underlying_foo(3)
    4
    """

    def _add_egress(func):
        @wraps(func)
        def func_with_added_egress(*args, **kwargs):
            output = func(*args, **kwargs)
            return post(output)

        return func_with_added_egress

    return _add_egress
