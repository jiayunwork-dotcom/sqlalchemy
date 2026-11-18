"""Tests for ordered ORM bulk UPDATE by primary key with RETURNING, i.e.
``Session.execute(update(Entity).returning(...,
sort_by_parameter_order=True), [list of mappings],
execution_options={"synchronize_session": False})``.

"""
from __future__ import annotations

import math
from typing import Optional

from sqlalchemy import bindparam
from sqlalchemy import delete
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import testing
from sqlalchemy import update
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_
from sqlalchemy.testing.entities import ComparableEntity
from sqlalchemy.testing.fixtures import fixture_session


def _make_models(decl_base):
    """Build a small model hierarchy used by several tests.

    * ``A`` is a single table entity (or the base of a joined inheritance
      hierarchy).
    * ``B`` is a joined-inheritance subclass of ``A``.
    * ``C`` has a composite primary key and/or renamed columns.
    """

    class A(ComparableEntity, decl_base):
        __tablename__ = "a"
        id: Mapped[int] = mapped_column(
            primary_key=True, autoincrement=False
        )
        x: Mapped[int]
        y: Mapped[Optional[str]]

    class B(A):
        __tablename__ = "b"
        id: Mapped[int] = mapped_column(
            ForeignKey(A.id), primary_key=True, autoincrement=False
        )
        z: Mapped[Optional[str]]

    class C(ComparableEntity, decl_base):
        __tablename__ = "c"
        key1: Mapped[int] = mapped_column(
            "k1", primary_key=True, autoincrement=False
        )
        key2: Mapped[str] = mapped_column("k2", primary_key=True)
        val: Mapped[int] = mapped_column("v_col")
        name: Mapped[str] = mapped_column("n_col")

    return A, B, C


class _CountExecutions:
    """count before_cursor_execute events; each execute/executemany
    invocation counts once."""

    def __init__(self, engine):
        self.count = 0
        self.engine = engine

    def __enter__(self):
        @event.listens_for(self.engine, "before_cursor_execute")
        def bce(conn, cursor, statement, parameters, context, executemany):
            self.count += 1

        self._fn = bce
        return self

    def __exit__(self, *exc):
        event.remove(self.engine, "before_cursor_execute", self._fn)


class OrderedBulkUpdateTest(fixtures.TestBase):
    __sparse_driver_backend__ = True

    @testing.fixture
    def models(self, decl_base):
        return _make_models(decl_base)

    @testing.fixture
    def single_table_data(self, models, decl_base):
        A, B, C = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=i, x=i * 10, y=f"y{i}") for i in range(1, 6)])
        s.commit()
        return s

    @testing.fixture
    def joined_inh_data(self, models, decl_base):
        A, B, C = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all(
            [B(id=i, x=i * 10, y=f"y{i}", z=f"z{i}") for i in range(1, 6)]
        )
        s.commit()
        return s

    def test_entity_returning_input_order(self, models, single_table_data):
        """out-of-order keys are returned in input order; missing keys are
        skipped"""
        A, _, _ = models
        s = single_table_data
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [
                {"id": 3, "x": 300},
                {"id": 1, "x": 100, "y": "yy1"},
                {"id": 99, "x": 999},
                {"id": 2, "y": "yy2"},
            ],
            execution_options={"synchronize_session": False},
        )
        eq_([a.id for a in result.scalars().all()], [3, 1, 2])
        eq_(
            s.execute(select(A.id, A.x, A.y).order_by(A.id)).all(),
            [
                (1, 100, "yy1"),
                (2, 20, "yy2"),
                (3, 300, "y3"),
                (4, 40, "y4"),
                (5, 50, "y5"),
            ],
        )

    def test_entity_returning_keys(self, models, single_table_data):
        A, _, _ = models
        s = single_table_data
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [{"id": 2, "x": 200}, {"id": 1, "x": 100}],
            execution_options={"synchronize_session": False},
        )
        eq_(list(result.keys()), ["A"])
        rows = result.all()
        eq_([a.id for a, in rows], [2, 1])

    def test_column_returning_scalars_mappings_keys(
        self, models, single_table_data
    ):
        A, _, _ = models
        s = single_table_data
        stmt = update(A).returning(
            A.x, A.y.label("y_label"), sort_by_parameter_order=True
        )
        result = s.execute(
            stmt,
            [{"id": 2, "x": 200}, {"id": 1, "y": "yy1"}],
            execution_options={"synchronize_session": False},
        )
        eq_(list(result.keys()), ["x", "y_label"])
        eq_(
            result.mappings().all(),
            [
                {"x": 200, "y_label": "y2"},
                {"x": 10, "y_label": "yy1"},
            ],
        )

        # scalars() yields the first column
        result = s.execute(
            stmt,
            [{"id": 2, "x": 222}],
            execution_options={"synchronize_session": False},
        )
        eq_(result.scalars().all(), [222])

    def test_heterogeneous_assignments_and_nulls(
        self, models, single_table_data
    ):
        """different rows may set different columns; explicit None becomes
        NULL; omitted columns keep their values"""
        A, _, _ = models
        s = single_table_data
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [
                {"id": 1, "x": 100, "y": "yy1"},
                {"id": 2, "y": None},
                {"id": 3, "x": 300},
            ],
            execution_options={"synchronize_session": False},
        )
        eq_(
            [(a.id, a.x, a.y) for a in result.scalars().all()],
            [(1, 100, "yy1"), (2, 20, None), (3, 300, "y3")],
        )
        eq_(
            s.execute(select(A.id, A.x, A.y).order_by(A.id)).all(),
            [(1, 100, "yy1"), (2, 20, None), (3, 300, "y3"),
             (4, 40, "y4"), (5, 50, "y5")],
        )

    def test_where_filtering_with_bindparam(self, models, single_table_data):
        """statement WHERE criteria with per-row bind parameters filter which
        keys get updated"""
        A, _, _ = models
        s = single_table_data
        stmt = (
            update(A)
            .where(A.x < bindparam("thresh"))
            .returning(A, sort_by_parameter_order=True)
        )
        result = s.execute(
            stmt,
            [
                {"id": 1, "x": 5, "thresh": 100},  # 10 < 100 -> update
                {"id": 2, "x": 6, "thresh": 15},  # 20 < 15 -> skip
                {"id": 3, "x": 7, "thresh": 100},  # 30 < 100 -> update
            ],
            execution_options={"synchronize_session": False},
        )
        eq_([a.id for a in result.scalars().all()], [1, 3])
        eq_(
            s.execute(select(A.id, A.x).order_by(A.id)).all(),
            [(1, 5), (2, 20), (3, 7), (4, 40), (5, 50)],
        )

    def test_where_constant_expression(self, models, single_table_data):
        A, _, _ = models
        s = single_table_data
        stmt = (
            update(A)
            .where(A.x > 15)
            .returning(A, sort_by_parameter_order=True)
        )
        result = s.execute(
            stmt,
            [{"id": i, "x": i * 100} for i in [4, 2, 1, 3, 5]],
            execution_options={"synchronize_session": False},
        )
        # id 1 has x=10 which is not > 15, so it is skipped
        eq_([a.id for a in result.scalars().all()], [4, 2, 3, 5])
        eq_(
            s.execute(select(A.id, A.x).order_by(A.id)).all(),
            [(1, 10), (2, 200), (3, 300), (4, 400), (5, 500)],
        )

    def test_composite_pk_and_renamed_columns(self, models, decl_base):
        """composite primary keys match on the full identity; attribute names
        may differ from column names; keys that would collide when flattened
        to strings are not confused"""
        _, _, C = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all(
            [
                C(key1=1, key2="a", val=10, name="n1a"),
                C(key1=1, key2="b", val=20, name="n1b"),
                C(key1=11, key2="a", val=30, name="n11a"),
            ]
        )
        s.commit()

        stmt = update(C).returning(C, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [
                {"key1": 11, "key2": "a", "val": 300},
                {"key1": 1, "key2": "a", "val": 100},
                {"key1": 1, "key2": "b", "name": "NEW"},
            ],
            execution_options={"synchronize_session": False},
        )
        eq_(
            [(c.key1, c.key2, c.val, c.name) for c in result.scalars().all()],
            [
                (11, "a", 300, "n11a"),
                (1, "a", 100, "n1a"),
                (1, "b", 20, "NEW"),
            ],
        )
        eq_(
            s.execute(
                select(C.key1, C.key2, C.val, C.name).order_by(C.key1, C.key2)
            ).all(),
            [
                (1, "a", 100, "n1a"),
                (1, "b", 20, "NEW"),
                (11, "a", 300, "n11a"),
            ],
        )

    def test_joined_inheritance(self, models, joined_inh_data):
        """base and sub tables agree on the matched key set; returned
        entities carry updated attributes from both tables"""
        _, B, _ = models
        s = joined_inh_data
        stmt = (
            update(B)
            .where(B.x < bindparam("thresh"))
            .returning(B, sort_by_parameter_order=True)
        )
        result = s.execute(
            stmt,
            [
                {"id": 3, "x": 5, "z": "zz3", "thresh": 100},  # match, both
                {"id": 1, "z": "zz1", "thresh": 5},  # no match -> sub skipped
                {"id": 2, "x": 7, "thresh": 100},  # match, base only
                {"id": 99, "x": 1, "thresh": 100},  # missing
            ],
            execution_options={"synchronize_session": False},
        )
        eq_(
            [(b.id, b.x, b.y, b.z) for b in result.scalars().all()],
            [(3, 5, "y3", "zz3"), (2, 7, "y2", "z2")],
        )
        eq_(
            s.execute(
                select(B.id, B.x, B.y, B.z).order_by(B.id)
            ).all(),
            [
                (1, 10, "y1", "z1"),
                (2, 7, "y2", "z2"),
                (3, 5, "y3", "zz3"),
                (4, 40, "y4", "z4"),
                (5, 50, "y5", "z5"),
            ],
        )

    def test_joined_inheritance_sub_only_shape(self, models, joined_inh_data):
        """a shape that only assigns sub-table columns still updates the sub
        table only for keys that match the base-table WHERE"""
        _, B, _ = models
        s = joined_inh_data
        stmt = (
            update(B)
            .where(B.x > 15)
            .returning(B, sort_by_parameter_order=True)
        )
        result = s.execute(
            stmt,
            [{"id": 1, "z": "zz1"}, {"id": 4, "z": "zz4"}],
            execution_options={"synchronize_session": False},
        )
        # id 1 has x=10 which is not > 15, so it is skipped entirely
        eq_([(b.id, b.z) for b in result.scalars().all()], [(4, "zz4")])
        eq_(
            s.execute(select(B.id, B.z).order_by(B.id)).all(),
            [(1, "z1"), (2, "z2"), (3, "z3"), (4, "zz4"), (5, "z5")],
        )

    def test_populate_existing(self, models, joined_inh_data):
        """with populate_existing=True, already-loaded objects are refreshed
        in place and the result contains the same objects; objects that were
        not matched are not touched"""
        _, B, _ = models
        s = fixture_session()
        obj4 = s.get(B, 4)
        obj5 = s.get(B, 5)
        eq_((obj4.x, obj4.z), (40, "z4"))

        stmt = update(B).returning(B, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [{"id": 4, "x": 444, "z": "zz4"}],
            execution_options={
                "synchronize_session": False,
                "populate_existing": True,
            },
        )
        returned = result.scalars().all()
        eq_(len(returned), 1)
        is_(returned[0], obj4)
        eq_((obj4.x, obj4.z), (444, "zz4"))
        # obj5 was not part of the update; unchanged
        eq_((obj5.x, obj5.z), (50, "z5"))

    def test_no_populate_existing_leaves_loaded_objects(
        self, models, joined_inh_data
    ):
        """without populate_existing, scalar results do not promise to
        refresh other already-loaded objects"""
        _, B, _ = models
        s = fixture_session()
        obj4 = s.get(B, 4)
        eq_(obj4.x, 40)
        stmt = update(B).returning(B, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [{"id": 4, "x": 444}],
            execution_options={"synchronize_session": False},
        )
        returned = result.scalars().all()
        eq_(len(returned), 1)
        # the already-present identity-map entry is not refreshed
        is_(returned[0], obj4)
        eq_(obj4.x, 40)
        # but the database row was updated
        eq_(s.execute(select(B.x).where(B.id == 4)).scalar(), 444)

    def test_empty_input(self, models, single_table_data):
        """an empty list of mappings produces an empty result that still
        carries the column metadata"""
        A, _, _ = models
        s = single_table_data
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        result = s.execute(
            stmt, [], execution_options={"synchronize_session": False}
        )
        eq_(result.all(), [])
        eq_(list(result.keys()), ["A"])

        stmt = update(A).returning(
            A.x, A.y.label("yy"), sort_by_parameter_order=True
        )
        result = s.execute(
            stmt, [], execution_options={"synchronize_session": False}
        )
        eq_(result.all(), [])
        eq_(list(result.keys()), ["x", "yy"])

    def test_rollback_undoes_writes(self, models, joined_inh_data):
        """writes to both the base and sub table are rolled back when the
        caller rolls back the transaction"""
        _, B, _ = models
        s = fixture_session()
        s.execute(
            update(B).returning(B, sort_by_parameter_order=True),
            [{"id": 1, "x": 111, "z": "zz1"}],
            execution_options={"synchronize_session": False},
        ).scalars().all()
        s.rollback()
        eq_(
            s.execute(select(B.id, B.x, B.z).order_by(B.id)).all(),
            [(1, 10, "z1"), (2, 20, "z2"), (3, 30, "z3"),
             (4, 40, "z4"), (5, 50, "z5")],
        )

    def test_constraint_failure_propagates_and_rolls_back(
        self, models, decl_base
    ):
        """a write error propagates and does not deliver a fake success
        result; the transaction is not committed"""
        A, _, _ = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=1, x=1, y="a"), A(id=2, x=2, y="b")])
        s.commit()

        with expect_raises_message(exc.IntegrityError, "NOT NULL"):
            s.execute(
                update(A).returning(A, sort_by_parameter_order=True),
                [{"id": 1, "x": None}],
                execution_options={"synchronize_session": False},
            ).scalars().all()
        s.rollback()
        eq_(
            s.execute(select(A.id, A.x, A.y).order_by(A.id)).all(),
            [(1, 1, "a"), (2, 2, "b")],
        )

    def test_paging_single_table_statement_count(self, models, decl_base):
        """large batches are paged; the number of cursor executions stays
        within 4*ceil(N/page_size)+2 for a single table"""
        A, _, _ = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        N, page = 200, 50
        s.add_all([A(id=i, x=i, y=f"y{i}") for i in range(N)])
        s.commit()

        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with _CountExecutions(testing.db) as counter:
            result = s.execute(
                stmt,
                [{"id": i, "x": i * 100} for i in range(N)],
                execution_options={
                    "synchronize_session": False,
                    "bulk_update_page_size": page,
                },
            )
            eq_(len(result.scalars().all()), N)
        limit = 4 * math.ceil(N / page) + 2
        assert counter.count <= limit, (
            f"expected at most {limit} cursor executions, "
            f"got {counter.count}"
        )
        eq_(
            s.execute(select(A.x).where(A.id == 0)).scalar(), 0
        )
        eq_(s.execute(select(A.x).where(A.id == 199)).scalar(), 19900)

    def test_paging_joined_inheritance_stmt_count(self, models, decl_base):
        """the number of cursor executions stays within
        6*ceil(N/page_size)+2 for two-level joined inheritance"""
        _, B, _ = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        N, page = 200, 50
        s.add_all([B(id=i, x=i, y=f"y{i}", z=f"z{i}") for i in range(N)])
        s.commit()

        stmt = update(B).returning(B, sort_by_parameter_order=True)
        with _CountExecutions(testing.db) as counter:
            result = s.execute(
                stmt,
                [{"id": i, "x": i * 100, "z": f"zz{i}"} for i in range(N)],
                execution_options={
                    "synchronize_session": False,
                    "bulk_update_page_size": page,
                },
            )
            eq_(len(result.scalars().all()), N)
        limit = 6 * math.ceil(N / page) + 2
        assert counter.count <= limit, (
            f"expected at most {limit} cursor executions, "
            f"got {counter.count}"
        )

    def test_paging_actually_pages(self, models, decl_base):
        """a small page size splits the batch into multiple UPDATE
        statements"""
        A, _, _ = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=i, x=i * 10, y=f"y{i}") for i in range(1, 8)])
        s.commit()

        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with _CountExecutions(testing.db) as counter:
            result = s.execute(
                stmt,
                [{"id": i, "x": i * 100} for i in [6, 2, 4, 1, 3, 5]],
                execution_options={
                    "synchronize_session": False,
                    "bulk_update_page_size": 2,
                },
            )
            eq_([a.id for a in result.scalars().all()], [6, 2, 4, 1, 3, 5])
        # 6 rows / page of 2 -> 3 UPDATE statements + 1 final SELECT
        eq_(counter.count, 4)


class OrderedBulkUpdateRejectionTest(fixtures.TestBase):
    __sparse_driver_backend__ = True

    @testing.fixture
    def models(self, decl_base):
        return _make_models(decl_base)

    def _session_with_data(self, models, decl_base):
        A, B, C = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=1, x=10, y="y1"), A(id=2, x=20, y="y2")])
        s.commit()
        return s

    def test_missing_pk(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "No primary key value supplied"
        ):
            s.execute(
                stmt,
                [{"x": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_null_pk(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "No primary key value supplied"
        ):
            s.execute(
                stmt,
                [{"id": None, "x": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_duplicate_pk(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "Duplicate primary key"
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": 1}, {"id": 1, "x": 2}],
                execution_options={"synchronize_session": False},
            )

    def test_unknown_attribute(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.ArgumentError, "Can't find mapped property named 'nope'"
        ):
            s.execute(
                stmt,
                [{"id": 1, "nope": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_no_assignment(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "at least one non-primary-key"
        ):
            s.execute(
                stmt,
                [{"id": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_where_param_is_not_an_assignment(self, models, decl_base):
        """a WHERE bind parameter key in the mapping is used for filtering
        and does not count as an assignment or unknown attribute"""
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = (
            update(A)
            .where(A.x < bindparam("thresh"))
            .returning(A, sort_by_parameter_order=True)
        )
        # this row only carries the pk and the where parameter; it has no
        # assignments and is therefore rejected
        with expect_raises_message(
            exc.InvalidRequestError, "at least one non-primary-key"
        ):
            s.execute(
                stmt,
                [{"id": 1, "thresh": 100}],
                execution_options={"synchronize_session": False},
            )

    def test_sql_expression_assignment(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "SQL expression assignments"
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": A.x + 1}],
                execution_options={"synchronize_session": False},
            )

    def test_statement_level_values(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = (
            update(A)
            .values(x=5)
            .returning(A, sort_by_parameter_order=True)
        )
        with expect_raises_message(
            exc.InvalidRequestError, "statement-level values"
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_relationship_assignment(self, models, decl_base):
        A, _, _ = models

        class R(ComparableEntity, decl_base):
            __tablename__ = "r"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            a_id: Mapped[int] = mapped_column(ForeignKey("a.id"))

        # give A a relationship; assigning to it in a bulk UPDATE mapping
        # is not supported
        A.rel = relationship(R)

        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError,
            "only plain mapped column assignments are supported",
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": 1, "rel": None}],
                execution_options={"synchronize_session": False},
            )

    def test_bad_page_size(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        for bad_page_size in (True, False, 0, -1, "50", 2.5, None):
            with expect_raises_message(
                exc.ArgumentError, "bulk_update_page_size"
            ):
                s.execute(
                    stmt,
                    [{"id": 1, "x": 1}],
                    execution_options={
                        "synchronize_session": False,
                        "bulk_update_page_size": bad_page_size,
                    },
                )

    def test_good_page_size(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        result = s.execute(
            stmt,
            [{"id": 1, "x": 1}],
            execution_options={
                "synchronize_session": False,
                "bulk_update_page_size": 5,
            },
        )
        eq_(len(result.scalars().all()), 1)

    @testing.variation("sync", ["auto", "evaluate", "fetch"])
    def test_sync_not_false(self, models, decl_base, sync):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        opts = {} if sync.auto else {"synchronize_session": sync.name}
        if sync.fetch:
            # 'fetch' is rejected for all bulk updates up front
            with expect_raises_message(
                exc.InvalidRequestError, "'fetch' synchronization"
            ):
                s.execute(stmt, [{"id": 1, "x": 1}], execution_options=opts)
        else:
            with expect_raises_message(
                exc.InvalidRequestError, "synchronize_session=False"
            ):
                s.execute(stmt, [{"id": 1, "x": 1}], execution_options=opts)

    def test_no_returning_clause(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "requires a RETURNING clause"
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": 1}],
                execution_options={"synchronize_session": False},
            )

    def test_version_id_col_not_supported(self, decl_base):
        class V(ComparableEntity, decl_base):
            __tablename__ = "v"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]
            version_id: Mapped[int] = mapped_column(nullable=False)
            __mapper_args__ = {"version_id_col": version_id}

        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add(V(id=1, x=1, version_id=1))
        s.commit()
        stmt = update(V).returning(V, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.InvalidRequestError, "version_id_col"
        ):
            s.execute(
                stmt,
                [{"id": 1, "x": 5}],
                execution_options={"synchronize_session": False},
            )

    def test_where_sub_table_column(self, models, decl_base):
        _, B, _ = models
        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add(B(id=1, x=10, y="y1", z="z1"))
        s.commit()
        stmt = (
            update(B)
            .where(B.z == "z1")
            .returning(B, sort_by_parameter_order=True)
        )
        with expect_raises_message(exc.InvalidRequestError, "base table"):
            s.execute(
                stmt,
                [{"id": 1, "x": 5}],
                execution_options={"synchronize_session": False},
            )

    def test_where_sql_function(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = (
            update(A)
            .where(func.lower(A.y) == "y1")
            .returning(A, sort_by_parameter_order=True)
        )
        with expect_raises_message(exc.InvalidRequestError, "unsupported"):
            s.execute(
                stmt,
                [{"id": 1, "x": 5}],
                execution_options={"synchronize_session": False},
            )

    def test_where_subquery(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = (
            update(A)
            .where(A.id.in_(select(A.id)))
            .returning(A, sort_by_parameter_order=True)
        )
        with expect_raises_message(exc.InvalidRequestError, "unsupported"):
            s.execute(
                stmt,
                [{"id": 1, "x": 5}],
                execution_options={"synchronize_session": False},
            )

    def test_core_connection_rejected(self, models, decl_base):
        A, _, _ = models
        decl_base.metadata.create_all(testing.db)
        stmt = update(A.__table__).returning(
            A.__table__.c.id, sort_by_parameter_order=True
        )
        with testing.db.connect() as conn:
            with expect_raises_message(
                exc.StatementError, "sort_by_parameter_order"
            ):
                conn.execute(stmt, {"id": 1, "x": 1})

    def test_core_executemany_rejected(self, models, decl_base):
        A, _, _ = models
        decl_base.metadata.create_all(testing.db)
        stmt = update(A.__table__).returning(
            A.__table__.c.id, sort_by_parameter_order=True
        )
        with testing.db.connect() as conn:
            with expect_raises_message(
                exc.StatementError, "sort_by_parameter_order"
            ):
                conn.execute(stmt, [{"id": 1, "x": 1}])

    def test_orm_single_statement_rejected(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = (
            update(A)
            .where(A.id == 1)
            .returning(A, sort_by_parameter_order=True)
        )
        with expect_raises_message(
            exc.StatementError, "sort_by_parameter_order"
        ):
            s.execute(stmt)

    def test_orm_single_dict_rejected(self, models, decl_base):
        A, _, _ = models
        s = self._session_with_data(models, decl_base)
        stmt = update(A).returning(A, sort_by_parameter_order=True)
        with expect_raises_message(
            exc.StatementError, "sort_by_parameter_order"
        ):
            s.execute(stmt, {"id": 1, "x": 1})


class OrderedBulkUpdateExpressionTest(fixtures.TestBase):
    """the sort_by_parameter_order flag is part of the Update expression
    and its cache identity"""

    __sparse_driver_backend__ = True

    def test_flag_preserved_and_cached(self, decl_base):
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]

        stmt = update(A).returning(A, sort_by_parameter_order=True)
        is_(stmt._sort_by_parameter_order, True)
        # preserved across where(), additional returning() and cloning
        is_(stmt.where(A.id > 5)._sort_by_parameter_order, True)
        is_(stmt.returning(A.x)._sort_by_parameter_order, True)
        is_(stmt._clone()._sort_by_parameter_order, True)

        # the flag is part of the cache identity
        stmt_no_flag = update(A).returning(A).where(A.id > 5)
        eq_(
            stmt.where(A.id > 5)._generate_cache_key()
            != stmt_no_flag._generate_cache_key(),
            True,
        )

    def test_insert_flag_unaffected(self, decl_base):
        """the INSERT sort_by_parameter_order parameter works as before"""
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]

        stmt = insert(A).returning(A, sort_by_parameter_order=True)
        is_(stmt._sort_by_parameter_order, True)

    def test_delete_flag_rejected(self, decl_base):
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]

        with expect_raises_message(
            exc.ArgumentError,
            "only applies to INSERT and UPDATE statements",
        ):
            delete(A).returning(A, sort_by_parameter_order=True)


class OrderedBulkUpdateBackcompatTest(fixtures.TestBase):
    """existing behaviors that must not change"""

    __sparse_driver_backend__ = True

    def test_default_bulk_update_still_works(self, decl_base):
        """the default (non-ordered) bulk UPDATE by primary key is
        unchanged"""
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]
            y: Mapped[str]

        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=i, x=i, y=f"y{i}") for i in range(1, 4)])
        s.commit()
        s.execute(
            update(A),
            [{"id": 1, "x": 100}, {"id": 2, "x": 200}],
            execution_options={"synchronize_session": False},
        )
        eq_(
            s.execute(select(A.id, A.x, A.y).order_by(A.id)).all(),
            [(1, 100, "y1"), (2, 200, "y2"), (3, 3, "y3")],
        )

    def test_single_statement_update_returning_still_works(self, decl_base):
        """a plain single-statement ORM UPDATE..RETURNING is unchanged"""
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]

        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        s.add_all([A(id=i, x=i) for i in range(1, 4)])
        s.commit()
        result = s.execute(
            update(A).where(A.x > 1).values(x=0).returning(A.id, A.x),
            execution_options={"synchronize_session": False},
        )
        eq_(sorted(result.all()), [(2, 0), (3, 0)])

    def test_insert_returning_still_works(self, decl_base):
        """bulk INSERT..RETURNING with sort_by_parameter_order is
        unchanged"""
        class A(ComparableEntity, decl_base):
            __tablename__ = "a"
            id: Mapped[int] = mapped_column(
                primary_key=True, autoincrement=False
            )
            x: Mapped[int]

        decl_base.metadata.create_all(testing.db)
        s = fixture_session()
        result = s.execute(
            insert(A).returning(A, sort_by_parameter_order=True),
            [{"id": 2, "x": 20}, {"id": 1, "x": 10}],
        )
        eq_([a.id for a in result.scalars().all()], [2, 1])
