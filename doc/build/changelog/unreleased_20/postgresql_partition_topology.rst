.. change::
    :tags: postgresql, feature

    Added the :paramref:`_schema.MetaData.reflect.postgresql_partition_topology`
    parameter to :meth:`_schema.MetaData.reflect`.  When enabled, a PostgreSQL
    declarative partitioning hierarchy selected by its root table is reflected
    as a connected set of :class:`_schema.Table` objects, rather than as a
    collection of unconnected tables.  Each reflected table exposes its place
    in the hierarchy via the new :attr:`_schema.Table.partition_parent`,
    :attr:`_schema.Table.partition_bound`,
    :attr:`_schema.Table.partition_is_default`,
    :attr:`_schema.Table.partition_key` and
    :attr:`_schema.Table.partition_strategy` attributes.

    The topology is carried through :meth:`_schema.Table.to_metadata`, so that
    a hierarchy may be copied into a new schema in any order;
    :attr:`_schema.MetaData.sorted_tables` then emits parents before their
    partitions, and both :meth:`_schema.MetaData.create_all` and single
    :class:`.CreateTable` constructs render child tables using
    ``CREATE TABLE ... PARTITION OF ... FOR VALUES ...``.

    The hierarchy is discovered using a single catalogue query regardless of
    its size, and is limited to ``RANGE`` and ``LIST`` partitioning over
    integer, date and text columns, within a single schema, with at most two
    levels of parent/child relationships.  Requests or hierarchies outside
    of that scope, including traditional ``INHERITS`` hierarchies and
    individual partitions named as roots, raise
    :class:`_exc.PartitionTopologyError` before any tables are added to the
    target :class:`_schema.MetaData`.
