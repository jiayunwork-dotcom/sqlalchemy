.. change::
    :tags: orm, feature
    :tickets: 13170

    Added support for ordered ORM bulk UPDATE by primary key with RETURNING.
    The :paramref:`_dml.UpdateBase.returning.sort_by_parameter_order`
    parameter, previously limited to :class:`_dml.Insert` statements, is now
    also accepted by :class:`_dml.Update` statements.  When an ORM-enabled
    :func:`_dml.update` construct that uses
    ``.returning(..., sort_by_parameter_order=True)`` is passed to
    :meth:`_orm.Session.execute` along with a list of parameter dictionaries
    (each carrying a full primary key), the batch is written using paged
    ``UPDATE ... FROM (VALUES ...)`` statements and the updated rows are
    delivered back through the usual ORM result and entity machinery in the
    original input order.  Batch sizing is controlled by the new
    ``bulk_update_page_size`` execution option.  The feature is currently
    supported for the SQLite dialect (SQLite 3.35+).  See
    :ref:`orm_queryguide_bulk_update_returning_ordered` for details.
