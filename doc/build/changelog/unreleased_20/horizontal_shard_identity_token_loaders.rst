.. change::
    :tags: bug, orm, ext

    Fixed several issues in the horizontal sharding extension where secondary
    loads triggered by objects that came from more than one shard did not
    respect the origin of those objects.  A relationship loader such as
    :func:`_orm.selectinload` or :func:`_orm.subqueryload`, a lazy loader,
    or the subclass column loader used by :func:`_orm.selectin_polymorphic`
    would use the full set of parent keys from every shard in the
    supplementary SELECT, and would emit that SELECT against every shard.
    When two shards held rows with the same primary key, each parent object
    could therefore be populated with the other shard's related rows.

    Such loaders now group the parent objects by the identity token each one
    was loaded under - the token being the only record of an object's
    origin that survives after the primary result has been consumed - and
    emit one supplementary SELECT per token.  Each such SELECT carries only
    the keys belonging to that token and is executed pinned to that token, so
    it reaches only the originating shard.  A parent row with no related
    rows on its own shard receives an empty collection (or ``None`` for a
    scalar relationship), regardless of what another shard holds under the
    same key.  Batching continues to apply within each token's group, so
    nested eager loaders and :meth:`_engine.Result.yield_per` streaming are
    partitioned the same way at every level rather than degrading to a query
    per object.

    An explicit shard selection - :class:`_horizontal.set_shard_id` or the
    ``shard_id`` bind argument - continues to constrain only the primary
    query; the loaders now follow the actual origin of the returned objects,
    which in that case is the one selected shard.  The identity map lookup
    performed for a many-to-one load is likewise limited to the parent row's
    own identity token when the parent is known, instead of consulting every
    shard returned by ``identity_chooser`` and potentially resolving to an
    unrelated row that shares the primary key.  As part of this change,
    ``yield_per`` and the processed ORM load options from the primary query
    are no longer re-applied to secondary loader SELECTs, which previously
    caused a sharded ``yield_per`` query with eager loaders to fail.
