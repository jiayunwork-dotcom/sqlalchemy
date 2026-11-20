"""Tests for per-shard propagation of secondary loads in the horizontal
sharding extension.

These tests target the behavior where a primary query spanning more than one
shard returns objects that share a primary key (and, in one case, a composite
primary key) across shards.  Secondary loaders - relationship eager loaders,
lazy loaders, and polymorphic subclass column loaders - must issue their
supplementary SELECTs only against the shard each parent row actually came
from, and each such SELECT must only carry the keys belonging to that shard.

"""

import os

from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.ext.horizontal_shard import set_shard_id
from sqlalchemy.ext.horizontal_shard import ShardedSession
from sqlalchemy.orm import lazyload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import selectin_polymorphic
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import subqueryload
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_
from sqlalchemy.testing import is_not
from sqlalchemy.testing.engines import testing_engine
from sqlalchemy.testing.engines import testing_reaper
from sqlalchemy.testing.provision import FOLLOWER_IDENT


class _Recorder:
    """record the statements emitted against one shard"""

    def __init__(self):
        self.statements = []

    def __call__(self, conn, cursor, statement, params, context, executemany):
        self.statements.append((statement, params))

    def clear(self):
        self.statements.clear()

    def keys(self):
        """the bound parameter values of each IN-style loader SELECT, which
        for these loaders is the flattened list of parent keys sent to the
        database
        """
        result = []
        for statement, params in self.statements:
            if " IN " in statement and params:
                result.append(tuple(params))
        return result


class ShardIdentityTest(fixtures.MappedTest):
    __requires__ = ("sqlite",)
    __skip_if__ = (lambda: False,)

    _n_pk = 1

    run_create_tables = None

    @classmethod
    def define_tables(cls, metadata):
        cls.tables.batch = Table(
            "batch",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(30), nullable=False),
            Column("kind", String(20), nullable=False),
        )

        cls.tables.big_batch = Table(
            "big_batch",
            metadata,
            Column("id", Integer, ForeignKey("batch.id"), primary_key=True),
            Column("extra", String(30)),
        )

        cls.tables.item = Table(
            "item",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("batch_id", Integer, ForeignKey("batch.id")),
            Column("label", String(30)),
        )

    @classmethod
    def setup_classes(cls):
        class Batch:
            pass

        class BigBatch(Batch):
            pass

        class Item:
            pass

        cls.classes.Batch = Batch
        cls.classes.BigBatch = BigBatch
        cls.classes.Item = Item

    @classmethod
    def setup_mappers(cls):
        Batch = cls.classes.Batch
        BigBatch = cls.classes.BigBatch
        Item = cls.classes.Item

        cls.mapper_registry.map_imperatively(
            Batch,
            cls.tables.batch,
            properties={
                "items": relationship(
                    Item, back_populates="batch", order_by="Item.id"
                )
            },
            polymorphic_on=cls.tables.batch.c.kind,
            polymorphic_identity="batch",
        )
        cls.mapper_registry.map_imperatively(
            BigBatch,
            cls.tables.big_batch,
            inherits=Batch,
            polymorphic_identity="big",
        )
        cls.mapper_registry.map_imperatively(
            Item,
            cls.tables.item,
            properties={"batch": relationship(Batch, back_populates="items")},
        )

    def _init_dbs(self):
        self.dbs = {}
        for shard in ("east", "west"):
            self.dbs[shard] = testing_engine(
                "sqlite:///shard_%s_%s.db" % (shard, FOLLOWER_IDENT)
            )

        for engine in self.dbs.values():
            self.tables_test_metadata.create_all(engine)

        # same primary keys on both shards, different values
        with self.dbs["east"].begin() as conn:
            conn.execute(
                self.tables.batch.insert(),
                [
                    {"id": 1, "name": "east-1", "kind": "big"},
                    {"id": 2, "name": "east-2", "kind": "batch"},
                ],
            )
            conn.execute(
                self.tables.big_batch.insert(),
                [{"id": 1, "extra": "east-extra"}],
            )
            conn.execute(
                self.tables.item.insert(),
                [
                    {"id": 10, "batch_id": 1, "label": "east-item-1"},
                    {"id": 11, "batch_id": 2, "label": "east-item-2"},
                ],
            )

        with self.dbs["west"].begin() as conn:
            conn.execute(
                self.tables.batch.insert(),
                [
                    {"id": 1, "name": "west-1", "kind": "big"},
                    {"id": 2, "name": "west-2", "kind": "batch"},
                ],
            )
            conn.execute(
                self.tables.big_batch.insert(),
                [{"id": 1, "extra": "west-extra"}],
            )
            conn.execute(
                self.tables.item.insert(),
                [{"id": 20, "batch_id": 1, "label": "west-item-1"}],
            )

        self.recorders = {shard: _Recorder() for shard in self.dbs}
        for shard, engine in self.dbs.items():
            event.listen(
                engine, "before_cursor_execute", self.recorders[shard]
            )

        return self.dbs

    def setup_test(self):
        self._init_dbs()

        Item = self.classes.Item

        def shard_chooser(mapper, instance, clause=None):
            if isinstance(instance, Item):
                return shard_chooser(mapper, instance.batch)
            return instance.name.split("-")[0]

        def identity_chooser(
            mapper,
            primary_key,
            *,
            lazy_loaded_from,
            execution_options,
            bind_arguments,
            **kw,
        ):
            return ["east", "west"]

        def execute_chooser(orm_context):
            return ["east", "west"]

        self.session = ShardedSession(
            shards=self.dbs,
            shard_chooser=shard_chooser,
            identity_chooser=identity_chooser,
            execute_chooser=execute_chooser,
        )

    def teardown_test(self):
        self.session.close()
        testing_reaper.checkin_all()
        for shard in self.dbs:
            os.remove("shard_%s_%s.db" % (shard, FOLLOWER_IDENT))

    def _clear(self):
        for recorder in self.recorders.values():
            recorder.clear()

    def _east_item_keys(self):
        # the second level prefetch loads Batch rows again, so these are the
        # parent keys of the batches the east items point at: both batches
        return (1, 2)

    def _west_item_keys(self):
        # the west items all point at batch 1 only
        return (1,)

    def _batch_keys(self):
        return (1, 2)

    def _assert_item_keys(self):
        eq_(self.recorders["east"].keys()[1], self._east_item_keys())
        eq_(self.recorders["west"].keys()[1], self._west_item_keys())

    def _load_batches(self, *options):
        Batch = self.classes.Batch
        self._clear()
        return (
            self.session.execute(select(Batch).options(*options))
            .unique()
            .scalars()
            .all()
        )

    def _assert_one_select_per_shard(self):
        # one relationship SELECT per shard; each carries every parent key
        # that came from that shard
        eq_(
            {
                shard: len(rec.statements)
                for shard, rec in self.recorders.items()
            },
            {"east": 2, "west": 2},
        )
        eq_(self.recorders["east"].keys(), [(1, 2)])
        eq_(self.recorders["west"].keys(), [(1, 2)])

    def test_selectinload_same_pk_partitioned_by_shard(self):
        Batch = self.classes.Batch

        batches = self._load_batches(selectinload(Batch.items))

        eq_(
            [(b.name, [i.label for i in b.items]) for b in batches],
            [
                ("east-1", ["east-item-1"]),
                ("east-2", ["east-item-2"]),
                ("west-1", ["west-item-1"]),
                ("west-2", []),
            ],
        )
        self._assert_one_select_per_shard()

        # a row that doesn't exist on a shard gives an empty collection
        eq_(batches[3].items, [])

    def test_subqueryload_same_pk_partitioned_by_shard(self):
        Batch = self.classes.Batch

        batches = self._load_batches(subqueryload(Batch.items))

        eq_(
            [(b.name, [i.label for i in b.items]) for b in batches],
            [
                ("east-1", ["east-item-1"]),
                ("east-2", ["east-item-2"]),
                ("west-1", ["west-item-1"]),
                ("west-2", []),
            ],
        )
        eq_(
            {
                shard: len(rec.statements)
                for shard, rec in self.recorders.items()
            },
            {"east": 2, "west": 2},
        )

    def test_lazyload_same_pk_partitioned_by_shard(self):
        Batch = self.classes.Batch

        batches = self._load_batches(lazyload(Batch.items))
        eq_(
            [(b.name, [i.label for i in b.items]) for b in batches],
            [
                ("east-1", ["east-item-1"]),
                ("east-2", ["east-item-2"]),
                ("west-1", ["west-item-1"]),
                ("west-2", []),
            ],
        )

    def test_lazy_many_to_one_uses_parent_shard(self):
        Item = self.classes.Item

        item = (
            self.session.execute(select(Item).filter(Item.id == 20))
            .scalars()
            .one()
        )
        self._clear()

        batch = item.batch
        eq_(batch.name, "west-1")
        eq_(batch.items[0].label, "west-item-1")
        eq_(self.recorders["east"].statements, [])
        # one SELECT for the many-to-one, one for the reverse collection,
        # both only on the shard the item came from
        eq_(len(self.recorders["west"].statements), 2)

    def test_lazy_many_to_one_uses_identity_map_only_on_parent_shard(self):
        Item = self.classes.Item

        # load both same-keyed batches so the identity map holds a "1" for
        # each shard, then expire the west item's relationship and reload it
        batches = self._load_batches()
        eq_(len(batches), 4)

        item = (
            self.session.execute(select(Item).filter(Item.id == 20))
            .scalars()
            .one()
        )
        self.session.expire(item, ["batch"])
        self._clear()

        # resolves to the west batch via the identity map, with no SQL at
        # all and no consultation of the east shard
        eq_(item.batch.name, "west-1")
        eq_(self.recorders["east"].statements, [])
        eq_(self.recorders["west"].statements, [])

    def test_polymorphic_subclass_loader_per_shard(self):
        Batch = self.classes.Batch
        BigBatch = self.classes.BigBatch

        batches = self._load_batches(selectin_polymorphic(Batch, [BigBatch]))

        eq_(
            [
                (b.name, type(b).__name__, getattr(b, "extra", None))
                for b in batches
            ],
            [
                ("east-1", "BigBatch", "east-extra"),
                ("east-2", "Batch", None),
                ("west-1", "BigBatch", "west-extra"),
                ("west-2", "Batch", None),
            ],
        )
        # exactly one subclass SELECT on each shard
        eq_(
            {
                shard: len(rec.statements)
                for shard, rec in self.recorders.items()
            },
            {"east": 2, "west": 2},
        )

    def test_explicit_shard_does_not_hit_other_shard(self):
        Batch = self.classes.Batch

        batches = self._load_batches(
            selectinload(Batch.items), set_shard_id("east")
        )

        eq_([b.name for b in batches], ["east-1", "east-2"])
        eq_(self.recorders["west"].statements, [])

    def test_explicit_shard_via_bind_argument(self):
        Batch = self.classes.Batch

        self._clear()
        batches = (
            self.session.execute(
                select(Batch)
                .options(selectinload(Batch.items))
                .where(Batch.id == 2),
                bind_arguments={"shard_id": "west"},
            )
            .scalars()
            .all()
        )

        eq_(
            [(b.name, [i.label for i in b.items]) for b in batches],
            [("west-2", [])],
        )
        eq_(self.recorders["east"].statements, [])

    def test_selectinload_with_yield_per_partitions_each_batch(self):
        Batch = self.classes.Batch

        self._clear()
        result = self.session.scalars(
            select(Batch)
            .options(selectinload(Batch.items))
            .execution_options(yield_per=1)
        )

        seen = []
        for batch in result:
            seen.append((batch.name, [i.label for i in batch.items]))

        eq_(
            seen,
            [
                ("east-1", ["east-item-1"]),
                ("east-2", ["east-item-2"]),
                ("west-1", ["west-item-1"]),
                ("west-2", []),
            ],
        )
        # one loader SELECT per yielded batch of parent rows, i.e. one per
        # parent row here, and each carries only that one parent's key
        for shard in ("east", "west"):
            rec = self.recorders[shard]
            eq_(
                [len(params) for _, params in rec.statements[1:]],
                [self._n_pk, self._n_pk],
            )

    def test_nested_selectinload_partitions_each_level(self):
        Batch = self.classes.Batch
        Item = self.classes.Item

        self._clear()
        batches = (
            self.session.execute(
                select(Batch).options(
                    selectinload(Batch.items).selectinload(Item.batch)
                )
            )
            .unique()
            .scalars()
            .all()
        )

        eq_(
            [(b.name, [i.label for i in b.items]) for b in batches],
            [
                ("east-1", ["east-item-1"]),
                ("east-2", ["east-item-2"]),
                ("west-1", ["west-item-1"]),
                ("west-2", []),
            ],
        )
        # each shard sees only the keys belonging to its own parent rows at
        # both levels of the prefetch
        eq_(len(self.recorders["east"].statements), 3)
        eq_(len(self.recorders["west"].statements), 3)
        # level one: both parent keys on each shard
        eq_(self.recorders["east"].keys()[0], self._batch_keys())
        eq_(self.recorders["west"].keys()[0], self._batch_keys())
        # level two: only the keys of the items on that shard; the west
        # shard only has one item so its second SELECT carries a single key
        self._assert_item_keys()

    def test_objects_keep_distinct_identity(self):
        Batch = self.classes.Batch
        Item = self.classes.Item

        batches = self._load_batches(selectinload(Batch.items))

        east_1 = [b for b in batches if b.name == "east-1"][0]
        west_1 = [b for b in batches if b.name == "west-1"][0]
        is_not(east_1, west_1)
        eq_(inspect(east_1).identity_key, (Batch, (1,), "east"))
        eq_(inspect(west_1).identity_key, (Batch, (1,), "west"))

        east_item = east_1.items[0]
        west_item = west_1.items[0]
        is_not(east_item, west_item)
        eq_(inspect(east_item).identity_key, (Item, (10,), "east"))
        eq_(inspect(west_item).identity_key, (Item, (20,), "west"))

        # both identities survive in the session
        is_(self.session.get(Batch, 1, identity_token="east"), east_1)
        is_(self.session.get(Batch, 1, identity_token="west"), west_1)


class ShardCompositeIdentityTest(ShardIdentityTest):
    """same scenario, but with a composite primary key"""

    _n_pk = 2

    def _east_item_keys(self):
        # the second level prefetch loads Batch rows again; east items point
        # at batches (1,1) and (1,2)
        return (1, 1, 1, 2)

    def _west_item_keys(self):
        # the west item points only at batch (1,1)
        return (1, 1)

    def _batch_keys(self):
        return (1, 1, 1, 2)

    @classmethod
    def define_tables(cls, metadata):
        cls.tables.batch = Table(
            "batch",
            metadata,
            Column("account_id", Integer, primary_key=True),
            Column("id", Integer, primary_key=True),
            Column("name", String(30), nullable=False),
            Column("kind", String(20), nullable=False),
        )

        cls.tables.big_batch = Table(
            "big_batch",
            metadata,
            Column("account_id", Integer, primary_key=True),
            Column("id", Integer, primary_key=True),
            Column("extra", String(30)),
            ForeignKeyConstraint(
                ["account_id", "id"], ["batch.account_id", "batch.id"]
            ),
        )

        cls.tables.item = Table(
            "item",
            metadata,
            Column("account_id", Integer, primary_key=True),
            Column("id", Integer, primary_key=True),
            Column("batch_account_id", Integer),
            Column("batch_id", Integer),
            Column("label", String(30)),
            ForeignKeyConstraint(
                ["batch_account_id", "batch_id"],
                ["batch.account_id", "batch.id"],
            ),
        )

    @classmethod
    def setup_mappers(cls):
        Batch = cls.classes.Batch
        BigBatch = cls.classes.BigBatch
        Item = cls.classes.Item

        cls.mapper_registry.map_imperatively(
            Batch,
            cls.tables.batch,
            properties={
                "items": relationship(
                    Item,
                    primaryjoin=(
                        cls.tables.batch.c.account_id
                        == cls.tables.item.c.batch_account_id
                    )
                    & (cls.tables.batch.c.id == cls.tables.item.c.batch_id),
                    back_populates="batch",
                    order_by="Item.account_id, Item.id",
                )
            },
            polymorphic_on=cls.tables.batch.c.kind,
            polymorphic_identity="batch",
        )
        cls.mapper_registry.map_imperatively(
            BigBatch,
            cls.tables.big_batch,
            inherits=Batch,
            polymorphic_identity="big",
        )
        cls.mapper_registry.map_imperatively(
            Item,
            cls.tables.item,
            properties={
                "batch": relationship(
                    Batch,
                    primaryjoin=(
                        cls.tables.batch.c.account_id
                        == cls.tables.item.c.batch_account_id
                    )
                    & (cls.tables.batch.c.id == cls.tables.item.c.batch_id),
                    back_populates="items",
                )
            },
        )

    def _init_dbs(self):
        self.dbs = {}
        for shard in ("east", "west"):
            self.dbs[shard] = testing_engine(
                "sqlite:///shard_%s_%s.db" % (shard, FOLLOWER_IDENT)
            )

        for engine in self.dbs.values():
            self.tables_test_metadata.create_all(engine)

        with self.dbs["east"].begin() as conn:
            conn.execute(
                self.tables.batch.insert(),
                [
                    {
                        "account_id": 1,
                        "id": 1,
                        "name": "east-1",
                        "kind": "big",
                    },
                    {
                        "account_id": 1,
                        "id": 2,
                        "name": "east-2",
                        "kind": "batch",
                    },
                ],
            )
            conn.execute(
                self.tables.big_batch.insert(),
                [{"account_id": 1, "id": 1, "extra": "east-extra"}],
            )
            conn.execute(
                self.tables.item.insert(),
                [
                    {
                        "account_id": 1,
                        "id": 10,
                        "batch_account_id": 1,
                        "batch_id": 1,
                        "label": "east-item-1",
                    },
                    {
                        "account_id": 1,
                        "id": 11,
                        "batch_account_id": 1,
                        "batch_id": 2,
                        "label": "east-item-2",
                    },
                ],
            )

        with self.dbs["west"].begin() as conn:
            conn.execute(
                self.tables.batch.insert(),
                [
                    {
                        "account_id": 1,
                        "id": 1,
                        "name": "west-1",
                        "kind": "big",
                    },
                    {
                        "account_id": 1,
                        "id": 2,
                        "name": "west-2",
                        "kind": "batch",
                    },
                ],
            )
            conn.execute(
                self.tables.big_batch.insert(),
                [{"account_id": 1, "id": 1, "extra": "west-extra"}],
            )
            conn.execute(
                self.tables.item.insert(),
                [
                    {
                        "account_id": 1,
                        "id": 20,
                        "batch_account_id": 1,
                        "batch_id": 1,
                        "label": "west-item-1",
                    }
                ],
            )

        self.recorders = {shard: _Recorder() for shard in self.dbs}
        for shard, engine in self.dbs.items():
            event.listen(
                engine, "before_cursor_execute", self.recorders[shard]
            )

        return self.dbs

    def _assert_one_select_per_shard(self):
        eq_(
            {
                shard: len(rec.statements)
                for shard, rec in self.recorders.items()
            },
            {"east": 2, "west": 2},
        )
        # the flattened IN parameter carries both key columns of each
        # parent row that came from the shard
        eq_(self.recorders["east"].keys(), [(1, 1, 1, 2)])
        eq_(self.recorders["west"].keys(), [(1, 1, 1, 2)])

    def test_objects_keep_distinct_identity(self):
        Batch = self.classes.Batch
        Item = self.classes.Item

        batches = self._load_batches(selectinload(Batch.items))

        east_1 = [b for b in batches if b.name == "east-1"][0]
        west_1 = [b for b in batches if b.name == "west-1"][0]
        is_not(east_1, west_1)
        eq_(inspect(east_1).identity_key, (Batch, (1, 1), "east"))
        eq_(inspect(west_1).identity_key, (Batch, (1, 1), "west"))

        east_item = east_1.items[0]
        west_item = west_1.items[0]
        is_not(east_item, west_item)
        eq_(inspect(east_item).identity_key, (Item, (1, 10), "east"))
        eq_(inspect(west_item).identity_key, (Item, (1, 20), "west"))

        is_(self.session.get(Batch, (1, 1), identity_token="east"), east_1)
        is_(self.session.get(Batch, (1, 1), identity_token="west"), west_1)


class ShardScalarRelationshipTest(fixtures.MappedTest):
    """a ``uselist=False`` relationship returns None for a shard that has no
    matching row, even when another shard has one with the same key
    """

    __requires__ = ("sqlite",)
    __skip_if__ = (lambda: False,)

    run_create_tables = None

    @classmethod
    def define_tables(cls, metadata):
        cls.tables.owner = Table(
            "owner",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(30), nullable=False),
            Column("detail_id", Integer),
        )
        cls.tables.detail = Table(
            "detail",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("value", String(30)),
        )

    @classmethod
    def setup_classes(cls):
        class Owner:
            pass

        class Detail:
            pass

        cls.classes.Owner = Owner
        cls.classes.Detail = Detail

    @classmethod
    def setup_mappers(cls):
        Owner = cls.classes.Owner
        Detail = cls.classes.Detail

        cls.mapper_registry.map_imperatively(
            Owner,
            cls.tables.owner,
            properties={
                "detail": relationship(
                    Detail,
                    primaryjoin=cls.tables.owner.c.detail_id
                    == cls.tables.detail.c.id,
                    foreign_keys=cls.tables.owner.c.detail_id,
                    uselist=False,
                )
            },
        )
        cls.mapper_registry.map_imperatively(Detail, cls.tables.detail)

    def setup_test(self):
        self.dbs = {}
        for shard in ("east", "west"):
            self.dbs[shard] = testing_engine(
                "sqlite:///shard_%s_%s.db" % (shard, FOLLOWER_IDENT)
            )
            self.tables_test_metadata.create_all(self.dbs[shard])

        # both shards have detail 1, but each owner points at a detail that
        # only exists on its own shard
        with self.dbs["east"].begin() as conn:
            conn.execute(
                self.tables.owner.insert(),
                [
                    {"id": 1, "name": "east-owner-1", "detail_id": 1},
                    {"id": 2, "name": "east-owner-2", "detail_id": 99},
                ],
            )
            conn.execute(
                self.tables.detail.insert(),
                [{"id": 1, "value": "east-detail-1"}],
            )

        with self.dbs["west"].begin() as conn:
            conn.execute(
                self.tables.owner.insert(),
                [
                    {"id": 1, "name": "west-owner-1", "detail_id": 1},
                    {"id": 2, "name": "west-owner-2", "detail_id": 1},
                ],
            )
            conn.execute(
                self.tables.detail.insert(),
                [{"id": 1, "value": "west-detail-1"}],
            )

        self.recorders = {}
        for shard, engine in self.dbs.items():
            recorder = _Recorder()
            self.recorders[shard] = recorder
            event.listen(engine, "before_cursor_execute", recorder)

        def shard_chooser(mapper, instance, clause=None):
            return instance.name.split("-")[0]

        self.session = ShardedSession(
            shards=self.dbs,
            shard_chooser=shard_chooser,
            identity_chooser=lambda *a, **k: ["east", "west"],
            execute_chooser=lambda ctx: ["east", "west"],
        )

    def teardown_test(self):
        self.session.close()
        testing_reaper.checkin_all()
        for shard in self.dbs:
            os.remove("shard_%s_%s.db" % (shard, FOLLOWER_IDENT))

    def test_scalar_relationship_resolves_per_shard(self):
        Owner = self.classes.Owner

        owners = (
            self.session.execute(
                select(Owner).options(selectinload(Owner.detail))
            )
            .scalars()
            .all()
        )

        eq_(
            [(o.name, o.detail.value if o.detail else None) for o in owners],
            [
                ("east-owner-1", "east-detail-1"),
                ("east-owner-2", None),
                ("west-owner-1", "west-detail-1"),
                ("west-owner-2", "west-detail-1"),
            ],
        )
        # although detail 1 exists on both shards, neither shard's loader
        # produced the other shard's detail row
        eq_(len(self.recorders["east"].statements), 2)
        eq_(len(self.recorders["west"].statements), 2)
