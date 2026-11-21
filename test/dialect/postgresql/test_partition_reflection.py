"""Tests for PostgreSQL partition hierarchy reflection.

These tests exercise the ``postgresql_partition_topology`` option of
:meth:`_schema.MetaData.reflect`, including the introspection attributes it
populates on :class:`_schema.Table`, the propagation of that topology through
:meth:`_schema.Table.to_metadata`, dependency-ordered DDL generation, and
round trip row routing into the recreated hierarchy.

"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import insert
from sqlalchemy import inspect
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import testing
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_
from sqlalchemy.testing import is_false
from sqlalchemy.testing import is_true


class PartitionTopologyFixtures:
    __only_on__ = "postgresql >= 10"
    __backend__ = True

    @testing.fixture
    def source_schema(self, connection):
        connection.exec_driver_sql('CREATE SCHEMA "topo_source"')
        try:
            yield "topo_source"
        finally:
            connection.exec_driver_sql('DROP SCHEMA "topo_source" CASCADE')

    @testing.fixture
    def target_schema(self, connection):
        connection.exec_driver_sql('CREATE SCHEMA "topo_target"')
        try:
            yield "topo_target"
        finally:
            connection.exec_driver_sql('DROP SCHEMA "topo_target" CASCADE')

    @testing.fixture
    def hierarchy(self, connection, source_schema):
        """A two level hierarchy: RANGE root over a LIST intermediate level.

        The hierarchy is::

            events                 RANGE (created)
            events_2024            LIST (kind), values a, b,c, d'e
            events_2024_rest       DEFAULT
            events_2025            LIST (kind)
            events_2025_a          values a
            events_2025_rest       DEFAULT
            events_before_2024     RANGE, MINVALUE .. 2024-01-01
            events_rest            DEFAULT

        """
        conn = connection
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events (
                id bigint NOT NULL,
                kind text NOT NULL,
                created date NOT NULL,
                note varchar(20),
                CONSTRAINT events_pkey PRIMARY KEY (id, created, kind)
            ) PARTITION BY RANGE (created)
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2024
                PARTITION OF topo_source.events
                FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
                PARTITION BY LIST (kind)
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2024_a
                PARTITION OF topo_source.events_2024
                FOR VALUES IN ('a', 'b,c', 'd''e')
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2024_rest
                PARTITION OF topo_source.events_2024 DEFAULT
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2025
                PARTITION OF topo_source.events
                FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')
                PARTITION BY LIST (kind)
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2025_a
                PARTITION OF topo_source.events_2025
                FOR VALUES IN ('a')
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_2025_rest
                PARTITION OF topo_source.events_2025 DEFAULT
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_before_2024
                PARTITION OF topo_source.events
                FOR VALUES FROM (MINVALUE) TO ('2024-01-01')
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.events_rest
                PARTITION OF topo_source.events DEFAULT
            """
        )
        return source_schema

    @testing.fixture
    def other_hierarchy(self, connection, source_schema):
        """A second, unrelated hierarchy in the same schema."""
        conn = connection
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.metrics (
                k integer NOT NULL,
                region text NOT NULL,
                value integer,
                CONSTRAINT metrics_pkey PRIMARY KEY (k, region)
            ) PARTITION BY RANGE (k)
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.metrics_lo
                PARTITION OF topo_source.metrics FOR VALUES FROM (0) TO (100)
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE topo_source.metrics_hi
                PARTITION OF topo_source.metrics FOR VALUES FROM (100) TO (200)
            """
        )
        # a plain table that is not part of any hierarchy
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.plain_table (id integer PRIMARY KEY)"
        )
        return source_schema


class ReflectionTest(PartitionTopologyFixtures, fixtures.TestBase):
    def test_reflects_selected_tree_only(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )

        eq_(
            set(meta.tables),
            {
                "topo_source.events",
                "topo_source.events_2024",
                "topo_source.events_2024_a",
                "topo_source.events_2024_rest",
                "topo_source.events_2025",
                "topo_source.events_2025_a",
                "topo_source.events_2025_rest",
                "topo_source.events_before_2024",
                "topo_source.events_rest",
            },
        )
        # the unrelated plain table is not pulled in
        assert "topo_source.plain_table" not in meta.tables

    def test_parent_linkage(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )

        root = meta.tables["topo_source.events"]
        mid = meta.tables["topo_source.events_2024"]
        leaf = meta.tables["topo_source.events_2024_a"]
        rest = meta.tables["topo_source.events_2024_rest"]
        direct = meta.tables["topo_source.events_before_2024"]

        is_(root.partition_parent, None)
        is_(mid.partition_parent, root)
        is_(leaf.partition_parent, mid)
        is_(rest.partition_parent, mid)
        is_(direct.partition_parent, root)

        is_true(rest.partition_is_default)
        is_false(leaf.partition_is_default)
        is_false(root.partition_is_default)

    def test_partition_keys_and_strategies(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )

        root = meta.tables["topo_source.events"]
        mid = meta.tables["topo_source.events_2024"]
        leaf = meta.tables["topo_source.events_2024_a"]

        eq_(root.partition_key, "RANGE (created)")
        eq_(root.partition_strategy, "RANGE")
        eq_(mid.partition_key, "LIST (kind)")
        eq_(mid.partition_strategy, "LIST")
        is_(leaf.partition_key, None)
        is_(leaf.partition_strategy, None)

    def test_partition_bounds(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )

        eq_(
            meta.tables["topo_source.events_2024"].partition_bound,
            "FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')",
        )
        eq_(
            meta.tables["topo_source.events_2024_a"].partition_bound,
            "FOR VALUES IN ('a', 'b,c', 'd''e')",
        )
        eq_(
            meta.tables["topo_source.events_before_2024"].partition_bound,
            "FOR VALUES FROM (MINVALUE) TO ('2024-01-01')",
        )
        eq_(
            meta.tables["topo_source.events_rest"].partition_bound,
            "DEFAULT",
        )
        is_(meta.tables["topo_source.events"].partition_bound, None)

    def test_root_columns_and_primary_key(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )
        root = meta.tables["topo_source.events"]
        eq_([c.name for c in root.columns], ["id", "kind", "created", "note"])
        eq_(
            [c.name for c in root.primary_key.columns],
            ["id", "created", "kind"],
        )
        eq_(root.primary_key.name, "events_pkey")

        # child tables carry the same column structure as the root
        for name in ("events_2024", "events_2024_a", "events_rest"):
            child = meta.tables["topo_source." + name]
            eq_(
                [c.name for c in child.columns],
                ["id", "kind", "created", "note"],
            )

    def test_quoted_identifiers_and_list_values(self, connection):
        connection.exec_driver_sql('CREATE SCHEMA "Odd Schema"')
        try:
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Schema"."Root Table" (
                    "Id" bigint NOT NULL,
                    "Kind Col" text NOT NULL,
                    "When" date NOT NULL,
                    CONSTRAINT "Root Table_pkey"
                        PRIMARY KEY ("Id", "Kind Col", "When")
                ) PARTITION BY RANGE ("When")
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Schema"."Child, One"
                    PARTITION OF "Odd Schema"."Root Table"
                    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
                    PARTITION BY LIST ("Kind Col")
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Schema"."Leaf A"
                    PARTITION OF "Odd Schema"."Child, One"
                    FOR VALUES IN ('x,y', 'it''s')
                """
            )

            meta = MetaData()
            meta.reflect(
                connection,
                schema="Odd Schema",
                only=["Root Table"],
                postgresql_partition_topology=True,
            )

            root = meta.tables["Odd Schema.Root Table"]
            # the raw, unquoted names are preserved
            eq_(root.name, "Root Table")
            eq_(root.partition_key, 'RANGE ("When")')
            eq_([c.name for c in root.columns], ["Id", "Kind Col", "When"])

            child = meta.tables["Odd Schema.Child, One"]
            eq_(child.name, "Child, One")
            is_(child.partition_parent, root)

            leaf = meta.tables["Odd Schema.Leaf A"]
            eq_(
                leaf.partition_bound,
                "FOR VALUES IN ('x,y', 'it''s')",
            )
        finally:
            connection.exec_driver_sql('DROP SCHEMA "Odd Schema" CASCADE')

    def test_multiple_trees(self, connection, hierarchy, other_hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["metrics", "events"],
            postgresql_partition_topology=True,
        )
        expected = {
            "topo_source.events",
            "topo_source.events_2024",
            "topo_source.events_2024_a",
            "topo_source.events_2024_rest",
            "topo_source.events_2025",
            "topo_source.events_2025_a",
            "topo_source.events_2025_rest",
            "topo_source.events_before_2024",
            "topo_source.events_rest",
            "topo_source.metrics",
            "topo_source.metrics_lo",
            "topo_source.metrics_hi",
        }
        eq_(set(meta.tables), expected)
        assert "topo_source.plain_table" not in meta.tables

    def test_child_names_not_required_in_only(self, connection, hierarchy):
        # naming a child in ``only`` alongside the root is harmless; the
        # root's descendants are loaded exactly once
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events", "events_2024"],
            postgresql_partition_topology=True,
        )
        eq_(
            set(meta.tables),
            {
                "topo_source.events",
                "topo_source.events_2024",
                "topo_source.events_2024_a",
                "topo_source.events_2024_rest",
                "topo_source.events_2025",
                "topo_source.events_2025_a",
                "topo_source.events_2025_rest",
                "topo_source.events_before_2024",
                "topo_source.events_rest",
            },
        )

    def test_root_without_children(self, connection, source_schema):
        connection.exec_driver_sql(
            "CREATE TABLE topo_source.empty_root (k int NOT NULL, v text) "
            "PARTITION BY LIST (k)"
        )
        meta = MetaData()
        meta.reflect(
            connection,
            schema=source_schema,
            only=["empty_root"],
            postgresql_partition_topology=True,
        )
        eq_(set(meta.tables), {"topo_source.empty_root"})
        root = meta.tables["topo_source.empty_root"]
        eq_(root.partition_key, "LIST (k)")
        is_(root.partition_parent, None)

    def test_integer_range_minvalue_maxvalue(self, connection, source_schema):
        connection.exec_driver_sql(
            "CREATE TABLE topo_source.byint ("
            "k int NOT NULL, v text, PRIMARY KEY (k)) PARTITION BY RANGE (k)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE topo_source.byint_lo "
            "PARTITION OF topo_source.byint FOR VALUES FROM (MINVALUE) TO (0)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE topo_source.byint_hi "
            "PARTITION OF topo_source.byint FOR VALUES FROM (0) TO (MAXVALUE)"
        )
        meta = MetaData()
        meta.reflect(
            connection,
            schema=source_schema,
            only=["byint"],
            postgresql_partition_topology=True,
        )
        eq_(
            meta.tables["topo_source.byint_lo"].partition_bound,
            "FOR VALUES FROM (MINVALUE) TO (0)",
        )
        eq_(
            meta.tables["topo_source.byint_hi"].partition_bound,
            "FOR VALUES FROM (0) TO (MAXVALUE)",
        )

    def test_subpartition_alone_rejected(self, connection, hierarchy):
        # a sub-partition named without its own root is not an acceptable
        # hierarchy root
        meta = MetaData()
        with expect_raises_message(
            exc.PartitionTopologyError,
            "is itself a partition of",
        ):
            meta.reflect(
                connection,
                schema=hierarchy,
                only=["events_2024"],
                postgresql_partition_topology=True,
            )
        eq_(len(meta.tables), 0)

    def test_repeat_reflect_is_idempotent(self, connection, hierarchy):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )
        before = dict(meta.tables)
        meta.reflect(
            connection,
            schema=hierarchy,
            only=["events"],
            postgresql_partition_topology=True,
        )
        eq_(set(meta.tables), set(before))
        for key, table in before.items():
            is_(meta.tables[key], table)


class ValidationTest(PartitionTopologyFixtures, fixtures.TestBase):
    def test_requires_schema(self, connection):
        with expect_raises_message(exc.ArgumentError, "schema"):
            MetaData().reflect(
                connection,
                only=["events"],
                postgresql_partition_topology=True,
            )

    def test_requires_only(self, connection, hierarchy):
        with expect_raises_message(exc.ArgumentError, "only"):
            MetaData().reflect(
                connection,
                schema=hierarchy,
                postgresql_partition_topology=True,
            )

    def test_requires_non_empty_only(self, connection, hierarchy):
        with expect_raises_message(exc.ArgumentError, "non-empty"):
            MetaData().reflect(
                connection,
                schema=hierarchy,
                only=[],
                postgresql_partition_topology=True,
            )

    def test_only_must_be_sequence(self, connection, hierarchy):
        with expect_raises_message(exc.ArgumentError, "only"):
            MetaData().reflect(
                connection,
                schema=hierarchy,
                only=lambda name, meta: True,
                postgresql_partition_topology=True,
            )

    def test_missing_root(self, connection, hierarchy):
        with expect_raises(exc.InvalidRequestError):
            MetaData().reflect(
                connection,
                schema=hierarchy,
                only=["no_such_table"],
                postgresql_partition_topology=True,
            )

    def test_top_level_plain_table(
        self, connection, hierarchy, other_hierarchy
    ):
        # a plain, non partitioned table is rejected as a hierarchy root
        meta = MetaData()
        with expect_raises_message(exc.PartitionTopologyError, "plain_table"):
            meta.reflect(
                connection,
                schema=hierarchy,
                only=["plain_table"],
                postgresql_partition_topology=True,
            )
        eq_(len(meta.tables), 0)


class UnsupportedHierarchyTest(PartitionTopologyFixtures, fixtures.TestBase):
    def _reflect(self, connection, schema, root):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=schema,
            only=[root],
            postgresql_partition_topology=True,
        )
        return meta

    def test_three_levels_rejected(self, connection, source_schema):
        from sqlalchemy.testing.assertions import expect_raises_message as erm

        conn = connection
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.deep (a int, b text) "
            "PARTITION BY RANGE (a)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.deep_1 PARTITION OF topo_source.deep "
            "FOR VALUES FROM (0) TO (10) PARTITION BY LIST (b)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.deep_1_x "
            "PARTITION OF topo_source.deep_1 FOR VALUES IN ('x') "
            "PARTITION BY RANGE (a)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.deep_1_x_lo "
            "PARTITION OF topo_source.deep_1_x FOR VALUES FROM (0) TO (5)"
        )
        with erm(exc.PartitionTopologyError, "two"):
            self._reflect(connection, source_schema, "deep")
        eq_(len(MetaData().tables), 0)

    def test_hash_partitioning_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.hashed (k int, v text) "
            "PARTITION BY HASH (k)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.hashed_0 "
            "PARTITION OF topo_source.hashed "
            "FOR VALUES WITH (MODULUS 2, REMAINDER 0)"
        )
        with expect_raises_message(exc.PartitionTopologyError, "RANGE"):
            self._reflect(connection, source_schema, "hashed")

    def test_cross_schema_descendant_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql('CREATE SCHEMA "topo_other"')
        try:
            conn.exec_driver_sql(
                "CREATE TABLE topo_source.crossroot (k date, v text) "
                "PARTITION BY RANGE (k)"
            )
            conn.exec_driver_sql(
                'CREATE TABLE "topo_other".crosschild '
                "PARTITION OF topo_source.crossroot "
                "FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')"
            )
            with expect_raises_message(
                exc.PartitionTopologyError, "different schema"
            ):
                self._reflect(connection, source_schema, "crossroot")
        finally:
            conn.exec_driver_sql('DROP SCHEMA "topo_other" CASCADE')

    def test_traditional_inheritance_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql("CREATE TABLE topo_source.parent_tbl (a int)")
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.child_tbl (b int) "
            "INHERITS (topo_source.parent_tbl)"
        )
        # neither the plain parent nor the inheritance child is a
        # declaratively partitioned table, so neither may be a root
        with expect_raises_message(
            exc.PartitionTopologyError, "not a partitioned table"
        ):
            self._reflect(connection, source_schema, "parent_tbl")
        with expect_raises_message(
            exc.PartitionTopologyError, "attached to a parent table"
        ):
            self._reflect(connection, source_schema, "child_tbl")

    def test_expression_key_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.exprkey (a int, b text) "
            "PARTITION BY RANGE ((a + 1))"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.exprkey_lo "
            "PARTITION OF topo_source.exprkey FOR VALUES FROM (0) TO (10)"
        )
        with expect_raises(exc.PartitionTopologyError):
            self._reflect(connection, source_schema, "exprkey")

    def test_unsupported_key_type_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.boolkey (k boolean, v text) "
            "PARTITION BY LIST (k)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.boolkey_t "
            "PARTITION OF topo_source.boolkey FOR VALUES IN (true)"
        )
        with expect_raises_message(exc.PartitionTopologyError, "type"):
            self._reflect(connection, source_schema, "boolkey")

    def test_multi_column_key_rejected(self, connection, source_schema):
        conn = connection
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.multikey ("
            "a int NOT NULL, b text NOT NULL, PRIMARY KEY (a, b)) "
            "PARTITION BY RANGE (a, b)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.multikey_1 "
            "PARTITION OF topo_source.multikey "
            "FOR VALUES FROM (0, 'a') TO (10, 'z')"
        )
        with expect_raises_message(
            exc.PartitionTopologyError, "single partition key column"
        ):
            self._reflect(connection, source_schema, "multikey")


class FailureAtomicityTest(PartitionTopologyFixtures, fixtures.TestBase):
    def test_unrelated_table_preserved(
        self, connection, hierarchy, other_hierarchy
    ):
        meta = MetaData()
        meta.reflect(connection, schema=hierarchy, only=["plain_table"])
        before = set(meta.tables)
        eq_(before, {"topo_source.plain_table"})

        with expect_raises(exc.PartitionTopologyError):
            meta.reflect(
                connection,
                schema=hierarchy,
                only=["plain_table"],
                postgresql_partition_topology=True,
            )
        eq_(set(meta.tables), before)
        is_(meta.tables["topo_source.plain_table"].partition_parent, None)

    def test_selected_tree_not_partially_added(self, connection, hierarchy):
        # make the hierarchy invalid deep inside: add an out-of-scope HASH
        # sub-hierarchy below a leaf that is otherwise valid
        conn = connection
        conn.exec_driver_sql("DROP TABLE topo_source.events_2024_a")
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.events_2024_a "
            "PARTITION OF topo_source.events_2024 "
            "FOR VALUES IN ('a', 'b,c', 'd''e') PARTITION BY HASH (id)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE topo_source.events_2024_a_0 "
            "PARTITION OF topo_source.events_2024_a "
            "FOR VALUES WITH (MODULUS 2, REMAINDER 0)"
        )

        meta = MetaData()
        with expect_raises(exc.PartitionTopologyError):
            meta.reflect(
                connection,
                schema=hierarchy,
                only=["events"],
                postgresql_partition_topology=True,
            )
        eq_(len(meta.tables), 0)

    def test_validation_precedes_loading(
        self, connection, hierarchy, other_hierarchy
    ):
        meta = MetaData()
        # request a valid root plus an invalid one; nothing may be added
        with expect_raises(exc.PartitionTopologyError):
            meta.reflect(
                connection,
                schema=hierarchy,
                only=["events", "plain_table"],
                postgresql_partition_topology=True,
            )
        eq_(len(meta.tables), 0)


class CopyAndDDLTest(PartitionTopologyFixtures, fixtures.TestBase):
    def _reflect(self, connection, schema, roots=("events",)):
        meta = MetaData()
        meta.reflect(
            connection,
            schema=schema,
            only=list(roots),
            postgresql_partition_topology=True,
        )
        return meta

    def test_sorted_tables_parents_first(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        # deliberately copy the children before the parent
        for name in [
            "events_2024_a",
            "events_2024_rest",
            "events_2025_a",
            "events_2025_rest",
            "events_rest",
            "events_before_2024",
            "events_2024",
            "events_2025",
            "events",
        ]:
            src.tables["topo_source." + name].to_metadata(
                dst, schema="topo_target"
            )

        names = [t.name for t in dst.sorted_tables]
        eq_(names[0], "events")
        assert names.index("events_2024") < names.index("events_2024_a")
        assert names.index("events_2024") < names.index("events_2024_rest")
        assert names.index("events_2025") < names.index("events_2025_a")
        assert names.index("events") < names.index("events_rest")

    def test_copied_parent_linkage_uses_new_schema(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        for name in reversed(list(src.tables)):
            src.tables[name].to_metadata(dst, schema="topo_target")

        root = dst.tables["topo_target.events"]
        mid = dst.tables["topo_target.events_2024"]
        leaf = dst.tables["topo_target.events_2024_a"]
        is_(root.partition_parent, None)
        is_(mid.partition_parent, root)
        is_(leaf.partition_parent, mid)
        eq_(
            leaf.partition_bound,
            "FOR VALUES IN ('a', 'b,c', 'd''e')",
        )
        eq_(mid.partition_key, "LIST (kind)")
        eq_(root.partition_strategy, "RANGE")

    def test_create_table_compilation(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        for name in src.tables:
            src.tables[name].to_metadata(dst, schema="topo_target")

        root_ddl = str(
            CreateTable(dst.tables["topo_target.events"]).compile(
                dialect=postgresql.dialect()
            )
        )
        assert "CREATE TABLE topo_target.events" in root_ddl
        assert "PARTITION BY RANGE (created)" in root_ddl
        assert "PRIMARY KEY (id, created, kind)" in root_ddl

        mid_ddl = str(
            CreateTable(dst.tables["topo_target.events_2024"]).compile(
                dialect=postgresql.dialect()
            )
        )
        assert (
            "PARTITION OF topo_target.events "
            "FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')" in mid_ddl
        )
        assert "PARTITION BY LIST (kind)" in mid_ddl

        leaf_ddl = str(
            CreateTable(dst.tables["topo_target.events_2024_a"]).compile(
                dialect=postgresql.dialect()
            )
        )
        assert "PARTITION OF topo_target.events_2024" in leaf_ddl
        assert "FOR VALUES IN ('a', 'b,c', 'd''e')" in leaf_ddl
        assert "PARTITION BY" not in leaf_ddl

    def test_create_all_and_row_routing(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        for name in src.tables:
            src.tables[name].to_metadata(dst, schema="topo_target")

        dst.create_all(connection)

        # actual parent linkage and bounds in the target schema
        rows = connection.exec_driver_sql(
            """
            SELECT c.relname,
                   pg_get_expr(c.relpartbound, c.oid),
                   (SELECT p.relname
                      FROM pg_inherits i
                      JOIN pg_class p ON p.oid = i.inhparent
                     WHERE i.inhrelid = c.oid)
              FROM pg_class c
             WHERE c.relnamespace = 'topo_target'::regnamespace
               AND c.relkind IN ('r', 'p')
            """
        ).all()
        got = {name: (bound, parent) for name, bound, parent in rows}
        eq_(got["events"], (None, None))
        eq_(
            got["events_2024"],
            ("FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')", "events"),
        )
        eq_(
            got["events_2024_a"],
            ("FOR VALUES IN ('a', 'b,c', 'd''e')", "events_2024"),
        )
        eq_(got["events_2024_rest"], ("DEFAULT", "events_2024"))
        eq_(got["events_rest"], ("DEFAULT", "events"))

        root = dst.tables["topo_target.events"]
        connection.execute(
            insert(root),
            [
                {"id": 1, "kind": "a", "created": "2024-06-01", "note": "a"},
                {"id": 2, "kind": "b,c", "created": "2024-06-02", "note": "b"},
                {"id": 3, "kind": "d'e", "created": "2024-06-03", "note": "c"},
                {"id": 4, "kind": "zzz", "created": "2024-07-01", "note": "d"},
                {"id": 5, "kind": "a", "created": "2025-03-01", "note": "e"},
                {"id": 6, "kind": "a", "created": "2023-01-01", "note": "f"},
                {"id": 7, "kind": "a", "created": "2026-01-01", "note": "g"},
            ],
        )

        def ids(table_name):
            return sorted(
                r.id
                for r in connection.execute(
                    select(dst.tables["topo_target." + table_name])
                )
            )

        eq_(ids("events_2024_a"), [1, 2, 3])
        eq_(ids("events_2024_rest"), [4])
        eq_(ids("events_2025_a"), [5])
        eq_(ids("events_before_2024"), [6])
        eq_(ids("events_rest"), [7])

    def test_create_without_source_schema_access(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        for name in src.tables:
            src.tables[name].to_metadata(dst, schema="topo_target")

        # drop the source schema entirely; creation must not need it
        connection.exec_driver_sql('DROP SCHEMA "topo_source" CASCADE')
        try:
            dst.create_all(connection)
            insp = inspect(connection)
            eq_(
                set(insp.get_table_names(schema="topo_target")),
                {
                    "events",
                    "events_2024",
                    "events_2024_a",
                    "events_2024_rest",
                    "events_2025",
                    "events_2025_a",
                    "events_2025_rest",
                    "events_before_2024",
                    "events_rest",
                },
            )
        finally:
            connection.exec_driver_sql('CREATE SCHEMA "topo_source"')

    def test_quoted_identifiers_round_trip(self, connection):
        connection.exec_driver_sql('CREATE SCHEMA "Odd Source"')
        connection.exec_driver_sql('CREATE SCHEMA "Odd Copy"')
        try:
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Source"."Root Table" (
                    "Id" bigint NOT NULL,
                    "Kind Col" text NOT NULL,
                    "When" date NOT NULL,
                    CONSTRAINT "Root Table_pkey"
                        PRIMARY KEY ("Id", "Kind Col", "When")
                ) PARTITION BY RANGE ("When")
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Source"."Child, One"
                    PARTITION OF "Odd Source"."Root Table"
                    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
                    PARTITION BY LIST ("Kind Col")
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Source"."Leaf A"
                    PARTITION OF "Odd Source"."Child, One"
                    FOR VALUES IN ('x,y', 'it''s')
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TABLE "Odd Source"."Leaf Default"
                    PARTITION OF "Odd Source"."Child, One" DEFAULT
                """
            )

            src = MetaData()
            src.reflect(
                connection,
                schema="Odd Source",
                only=["Root Table"],
                postgresql_partition_topology=True,
            )
            dst = MetaData()
            for key in reversed(list(src.tables)):
                src.tables[key].to_metadata(dst, schema="Odd Copy")

            dst.create_all(connection)

            root = dst.tables["Odd Copy.Root Table"]
            connection.execute(
                insert(root),
                [
                    {"Id": 1, "Kind Col": "x,y", "When": "2024-03-01"},
                    {"Id": 2, "Kind Col": "it's", "When": "2024-03-02"},
                    {"Id": 3, "Kind Col": "other", "When": "2024-03-03"},
                ],
            )

            def ids(name):
                return sorted(
                    r.Id
                    for r in connection.execute(
                        select(dst.tables["Odd Copy." + name])
                    )
                )

            eq_(ids("Leaf A"), [1, 2])
            eq_(ids("Leaf Default"), [3])

            rows = connection.exec_driver_sql(
                """
                SELECT c.relname,
                       pg_get_expr(c.relpartbound, c.oid),
                       (SELECT p.relname
                          FROM pg_inherits i
                          JOIN pg_class p ON p.oid = i.inhparent
                         WHERE i.inhrelid = c.oid)
                  FROM pg_class c
                 WHERE c.relnamespace = '"Odd Copy"'::regnamespace
                   AND c.relkind IN ('r', 'p')
                """
            ).all()
            got = {name: (bound, parent) for name, bound, parent in rows}
            eq_(
                got["Child, One"],
                (
                    "FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')",
                    "Root Table",
                ),
            )
            eq_(
                got["Leaf A"],
                ("FOR VALUES IN ('x,y', 'it''s')", "Child, One"),
            )
            eq_(got["Leaf Default"], ("DEFAULT", "Child, One"))
        finally:
            connection.exec_driver_sql('DROP SCHEMA "Odd Source" CASCADE')
            connection.exec_driver_sql('DROP SCHEMA "Odd Copy" CASCADE')

    def test_create_all_query_count(
        self, connection, hierarchy, target_schema
    ):
        src = self._reflect(connection, hierarchy)
        dst = MetaData()
        for name in reversed(list(src.tables)):
            src.tables[name].to_metadata(dst, schema="topo_target")

        statements = []

        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            statements.append(statement)

        event.listen(
            connection.engine, "before_cursor_execute", before_cursor_execute
        )
        try:
            dst.create_all(connection)
        finally:
            event.remove(
                connection.engine,
                "before_cursor_execute",
                before_cursor_execute,
            )

        # exactly one CREATE TABLE per table, plus a bounded number of
        # existence probes (per hierarchy, not per table)
        create_statements = [
            s for s in statements if s.lstrip().startswith("CREATE TABLE")
        ]
        eq_(len(create_statements), len(dst.tables))
        assert len(statements) <= len(dst.tables) + 3


class QueryCountTest(PartitionTopologyFixtures, fixtures.TestBase):
    """The topology option adds a single backend query, regardless of size."""

    @testing.fixture
    def counter(self, connection):
        counts = {"n": 0}

        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            counts["n"] += 1

        event.listen(
            connection.engine, "before_cursor_execute", before_cursor_execute
        )
        try:
            yield counts
        finally:
            event.remove(
                connection.engine,
                "before_cursor_execute",
                before_cursor_execute,
            )

    def _build_tree(self, connection, schema, leaves):
        conn = connection
        conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        conn.exec_driver_sql(
            f'CREATE TABLE "{schema}".root ('
            "k int NOT NULL, part text NOT NULL, v text, "
            "PRIMARY KEY (k, part)) PARTITION BY RANGE (k)"
        )
        if leaves == 8:
            conn.exec_driver_sql(
                f'CREATE TABLE "{schema}".mid PARTITION OF "{schema}".root '
                "FOR VALUES FROM (0) TO (1000) PARTITION BY LIST (part)"
            )
            for i in range(7):
                conn.exec_driver_sql(
                    f'CREATE TABLE "{schema}".leaf{i} '
                    f'PARTITION OF "{schema}".mid '
                    f"FOR VALUES IN ('p{i}')"
                )
            conn.exec_driver_sql(
                f'CREATE TABLE "{schema}".leafdef '
                f'PARTITION OF "{schema}".mid DEFAULT'
            )
        else:
            for m in range(8):
                conn.exec_driver_sql(
                    f'CREATE TABLE "{schema}".mid{m} '
                    f'PARTITION OF "{schema}".root '
                    f"FOR VALUES FROM ({m * 1000}) TO ({(m + 1) * 1000}) "
                    "PARTITION BY LIST (part)"
                )
                for i in range(9):
                    conn.exec_driver_sql(
                        f'CREATE TABLE "{schema}".m{m}_leaf{i} '
                        f'PARTITION OF "{schema}".mid{m} '
                        f"FOR VALUES IN ('p{i}')"
                    )
                conn.exec_driver_sql(
                    f'CREATE TABLE "{schema}".m{m}_def '
                    f'PARTITION OF "{schema}".mid{m} DEFAULT'
                )
        return [
            r[0]
            for r in conn.exec_driver_sql(
                f"SELECT relname FROM pg_class "
                f"WHERE relnamespace = '\"{schema}\"'::regnamespace "
                f"AND relkind IN ('r', 'p') ORDER BY relname"
            ).all()
        ]

    def _count(self, connection, counter, schema, roots, topology):
        counter["n"] = 0
        meta = MetaData()
        meta.reflect(
            connection,
            schema=schema,
            only=roots,
            **({"postgresql_partition_topology": True} if topology else {}),
        )
        return counter["n"], len(meta.tables)

    def test_query_counts(self, connection, counter):
        conn = connection
        names8 = self._build_tree(conn, "topo_q8", 8)
        names80 = self._build_tree(conn, "topo_q80", 80)
        try:
            normal8, count8 = self._count(
                conn, counter, "topo_q8", names8, False
            )
            topo8, topo_count8 = self._count(
                conn, counter, "topo_q8", ["root"], True
            )
            normal80, count80 = self._count(
                conn, counter, "topo_q80", names80, False
            )
            topo80, topo_count80 = self._count(
                conn, counter, "topo_q80", ["root"], True
            )

            # same table sets are loaded
            eq_(count8, topo_count8)
            eq_(count80, topo_count80)

            # the topology discovery is a single additional round trip,
            # independent of the number of tables
            eq_(topo8 - normal8, 1)
            eq_(topo80 - normal80, 1)
            assert topo80 - topo8 <= 4
        finally:
            conn.exec_driver_sql('DROP SCHEMA "topo_q8" CASCADE')
            conn.exec_driver_sql('DROP SCHEMA "topo_q80" CASCADE')


class DefaultBehaviourTest(PartitionTopologyFixtures, fixtures.TestBase):
    def test_plain_reflect_unchanged(self, connection, hierarchy):
        # without the option, a plain reflection of a child table yields
        # an unconnected Table with no topology attributes
        meta = MetaData()
        meta.reflect(connection, schema=hierarchy, only=["events"])
        eq_(set(meta.tables), {"topo_source.events"})
        root = meta.tables["topo_source.events"]
        is_(root.partition_parent, None)
        is_(root.partition_bound, None)
        is_(root.partition_key, None)
        # and the reflected table can still be created
        ddl = str(CreateTable(root).compile(dialect=postgresql.dialect()))
        assert "CREATE TABLE topo_source.events" in ddl
