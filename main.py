"""
Weaviate v4 async handler implementing BaseVectorDatabaseHandler.

Architecture notes
------------------
* The ``WeaviateAsyncClient`` is created in ``__init__`` but its network
  connection is deferred to ``initialize()``, which calls ``await
  client.connect()``.  The call is idempotent (the SDK no-ops if already
  connected), so it is safe to call from a FastAPI lifespan without extra
  guards.

* ``collections.get(name)`` is **synchronous** (zero-cost, just builds a
  Python wrapper) so it is called inline without ``await``.

* Every public method that hits the network is ``async`` and awaits the SDK
  coroutine directly — no ``asyncio.run``, no thread-pool executors.  This
  keeps the handler fully compatible with a FastAPI async request handler.

* Tenant isolation mirrors the Qdrant approach: a ``tenant_id`` property is
  stored on every object and injected into every filter.  Weaviate's native
  multi-tenancy is intentionally avoided to keep the schema identical across
  vector-backend swaps.

* Vector dimensions are **not** exposed in Weaviate's schema config (unlike
  Qdrant).  The embedder name is instead written as a dedicated property
  (``embedder_name``) on every object and checked by sampling one point.  An
  empty collection is always considered compatible.

* Score semantics follow Qdrant convention (higher = more similar).
  Weaviate returns cosine *distance* (lower = better), so the handler converts
  with ``score = 1.0 - distance``.

* ``get_all_tenant_points_from_files`` uses ``NOT(source like 'http*') AND
  source != ''`` so that objects without a source field are excluded, matching
  the intent of the Qdrant original (which filtered by a non-empty source that
  does not start with "http").

* Cursor-based pagination: Weaviate uses an ``after`` UUID cursor rather than
  a numeric offset.  The ``next_offset`` returned by the scroll helpers is the
  UUID string of the last returned object (or ``None`` when exhausted).
  Callers that pass an integer offset from the Qdrant path will receive a
  ``TypeError``; they must be migrated to UUID cursors when switching backends.

* Batch insert uses ``data.insert_many()`` which is a single gRPC call and is
  materially faster than looping ``data.insert()``.

* ``save_dump`` triggers Weaviate's filesystem-backup module.  The server must
  be started with ``ENABLE_MODULES=backup-filesystem`` and
  ``BACKUP_FILESYSTEM_PATH`` configured.
"""
import json
import uuid
from typing import Any, Dict, Iterable, List, Tuple, Type
import weaviate
import weaviate.classes as wvc
from langchain_core.documents import Document as LangChainDocument
from pydantic import ConfigDict
from weaviate.backup import BackupStorage
from weaviate.classes.query import Filter as WFilter, MetadataQuery
from weaviate.collections.classes.filters import _Filters as WeaviateFilter

from cat import BaseVectorDatabaseHandler, VectorDatabaseSettings, log
from cat.env import get_env
from cat.services.memory.models import (
    DocumentRecall,
    PointStruct,
    Record,
    ScoredPoint,
    UpdateResult,
    VectorMemoryType,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col(name: str) -> str:
    """Weaviate collection names must start with an upper-case letter."""
    return name[0].upper() + name[1:] if name else name


def _tenant_filter(agent_id: str) -> WeaviateFilter:
    return WFilter.by_property("tenant_id").equal(agent_id)


def _combine(*filters: WeaviateFilter) -> WeaviateFilter:
    """Combine one or more filters with AND.  Single filter is returned as-is."""
    valid = [f for f in filters if f is not None]
    if not valid:
        raise ValueError("At least one filter is required")
    return WFilter.all_of(valid) if len(valid) > 1 else valid[0]


def _meta_props(content: str, metadata: Dict | None, agent_id: str, embedder_name: str = "") -> Dict:
    """Build the flat Weaviate property dict for an object."""
    meta = metadata or {}
    return {
        "page_content": content,
        "tenant_id": agent_id,
        "metadata_json": json.dumps(meta),
        "metadata__source": meta.get("source", ""),
        "embedder_name": embedder_name,
    }


def _obj_to_record(obj) -> Record:
    """Convert a Weaviate QueryObject to a generic Record."""
    props = obj.properties or {}

    raw_meta = props.get("metadata_json", "{}")
    try:
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
    except (json.JSONDecodeError, TypeError):
        meta = {}

    # Weaviate v4 wraps single-vector responses in {"default": [...]}
    vec = None
    if obj.vector:
        vec = obj.vector.get("default") if isinstance(obj.vector, dict) else list(obj.vector)

    uid = str(obj.uuid)
    return Record(
        id=uid,
        payload={
            "id": uid,
            "page_content": props.get("page_content", ""),
            "metadata": meta,
            "tenant_id": props.get("tenant_id", ""),
        },
        vector=vec,
    )


def _obj_to_scored_point(obj, score: float = 0.0) -> ScoredPoint:
    r = _obj_to_record(obj)
    return ScoredPoint(**r.model_dump(), score=score)


def _record_to_doc_recall(point: Record | ScoredPoint) -> DocumentRecall:
    payload = point.payload or {}
    page_content = payload.get("page_content", "")
    if isinstance(page_content, dict):
        page_content = json.dumps(page_content)
    meta = payload.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    doc = DocumentRecall(
        document=LangChainDocument(
            page_content=page_content,
            metadata=meta,
            id=str(point.id),
        ),
        vector=point.vector,
        id=str(point.id),
    )
    if isinstance(point, ScoredPoint):
        doc.score = point.score
    return doc


def _distance_to_score(obj) -> float:
    """Convert Weaviate cosine distance to similarity score (Qdrant convention)."""
    dist = float(obj.metadata.distance) if (obj.metadata and obj.metadata.distance is not None) else None
    return (1.0 - dist) if dist is not None else 0.0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class WeaviateHandler(BaseVectorDatabaseHandler):
    """
    Async Weaviate handler compatible with BaseVectorDatabaseHandler.

    Args:
        host: Hostname of the Weaviate server (e.g. ``"localhost"``).
        http_port: HTTP/REST port (default ``8080``).
        grpc_port: gRPC port (default ``50051``).
        api_key: Optional API key for authenticated deployments.
        save_memory_snapshots: Trigger filesystem backups before destructive ops.
    """
    def __init__(
        self,
        host: str,
        http_port: int = 8080,
        grpc_port: int = 50051,
        api_key: str | None = None,
        save_memory_snapshots: bool = False,
    ):
        if not host:
            raise ValueError("CAT_WEAVIATE_HOST is not set.")

        super().__init__(save_memory_snapshots)
        self.host = host
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.api_key = api_key or None

        auth = wvc.init.Auth.api_key(self.api_key) if self.api_key else None

        # Client is created here but not yet connected.
        # Connection is established in initialize(), which is called from
        # the FastAPI lifespan (or equivalent startup hook).
        self._client: weaviate.WeaviateAsyncClient = weaviate.use_async_with_custom(
            http_host=host,
            http_port=http_port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=auth,
            skip_init_checks=False,
        )

    # ------------------------------------------------------------------
    # Equality / identity
    # ------------------------------------------------------------------

    def _eq(self, other: "WeaviateHandler") -> bool:
        return (
            self.host == other.host
            and self.http_port == other.http_port
            and self.grpc_port == other.grpc_port
        )

    @property
    def client(self) -> weaviate.WeaviateAsyncClient:
        return self._client

    def is_db_remote(self) -> bool:
        return True

    async def connect(self):
        if not self._client.is_connected():
            await self._client.connect()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, embedder_name: str, embedder_size: int):
        """
        Connect to Weaviate and ensure all memory collections exist and have
        the correct embedder. Called once from the FastAPI lifespan; safe to
        call again (connect() is idempotent).
        """
        await self.connect()

        for raw_name in self._collection_names:
            cname = _col(raw_name)
            exists = await self.check_collection_existence(cname)
            if exists:
                same = await self._check_embedding_size(embedder_name, embedder_size, cname)
                if same:
                    continue
                if self.save_memory_snapshots:
                    await self.save_dump(cname)
                await self.delete_collection(cname)
            await self.create_collection(embedder_name, embedder_size, cname)

    async def close(self):
        if self._client is not None and self._client.is_connected():
            await self._client.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def get_collection_names(self) -> List[str]:
        return list((await self._client.collections.list_all(simple=True)).keys())

    async def check_collection_existence(self, collection_name: str) -> bool:
        return await self._client.collections.exists(collection_name)

    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        log.warning(f"Creating Weaviate collection `{collection_name}` for agent `{self.agent_id}`...")
        try:
            await self._client.collections.create(
                name=collection_name,
                description=f"embedder:{embedder_name}",  # used for change detection
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE,
                    # Scalar quantisation mirrors Qdrant INT8 / always_ram=True.
                    quantizer=wvc.config.Configure.VectorIndex.Quantizer.sq(),
                ),
                properties=[
                    wvc.config.Property(name="page_content",     data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="tenant_id",        data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="metadata_json",    data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="metadata__source", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="embedder_name",    data_type=wvc.config.DataType.TEXT),
                ],
            )
            log.warning(f"Collection `{collection_name}` created for agent `{self.agent_id}`.")
        except Exception as exc:
            log.error(f"Error creating collection `{collection_name}`: {exc}")
            raise

    async def create_hybrid_collection(
        self,
        collection_name: str,
        dense_vector_config_name: str,
        sparse_vector_config_name: str,
    ):
        """
        Weaviate natively fuses BM25+vector on any collection; no separate
        sparse-vector collection is needed.  This method is a no-op when the
        collection already exists; otherwise it delegates to create_collection.
        """
        if await self.check_collection_existence(collection_name):
            return

        # Borrow the embedder name from the declarative collection description.
        declarative_cname = _col(str(VectorMemoryType.DECLARATIVE))
        embedder_name = "hybrid"
        try:
            col = self._client.collections.get(declarative_cname)
            cfg = await col.config.get(simple=True)
            if cfg.description and cfg.description.startswith("embedder:"):
                embedder_name = cfg.description[len("embedder:"):]
        except Exception:
            pass

        await self.create_collection(embedder_name, 0, collection_name)

    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        await self._client.collections.delete(collection_name)
        log.warning(f"Collection `{collection_name}` deleted for agent `{self.agent_id}`.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _check_embedding_size(
        self, embedder_name: str, embedder_size: int, collection_name: str
    ) -> bool:
        """
        Weaviate does not expose vector dimensions in the schema, so we sample
        one object and compare the stored ``embedder_name`` property.  An
        empty collection is treated as compatible.
        """
        try:
            col = self._client.collections.get(collection_name)
            result = await col.query.fetch_objects(
                filters=_tenant_filter(self.agent_id),
                limit=1,
                return_properties=["embedder_name"],
                include_vector=False,
            )
            if not result.objects:
                return True
            stored = result.objects[0].properties.get("embedder_name", "")
            if stored == embedder_name:
                log.debug(f"Collection `{collection_name}` has the same embedder.")
                return True
            log.warning(
                f"Collection `{collection_name}` embedder mismatch: "
                f"stored=`{stored}` requested=`{embedder_name}`."
            )
            return False
        except Exception as exc:
            log.warning(f"Could not verify embedder for `{collection_name}`: {exc}")
            return False

    def _build_filter(self, metadata: Dict | None = None) -> WeaviateFilter:
        """Tenant filter optionally AND-ed with arbitrary metadata conditions."""
        conditions: List[WeaviateFilter] = [_tenant_filter(self.agent_id)]
        if metadata:
            for key, value in metadata.items():
                conditions.extend(self.build_condition(key, value))
        return _combine(*conditions)

    # ------------------------------------------------------------------
    # Tenant filter condition (interface requirement)
    # ------------------------------------------------------------------

    def tenant_field_condition(self) -> WeaviateFilter:
        """Returns the Weaviate filter used to scope queries to this agent."""
        return _tenant_filter(self.agent_id)

    # ------------------------------------------------------------------
    # CRUD -- add points
    # ------------------------------------------------------------------

    async def add_point_to_tenant(
        self,
        collection_name: str,
        content: str,
        vector: Iterable,
        metadata: Dict = None,
        id_point: str | None = None,
        **kwargs,
    ) -> PointStruct | None:
        id_point = id_point or uuid.uuid4().hex
        col = self._client.collections.get(collection_name)
        props = _meta_props(content, metadata, self.agent_id)
        vec = list(vector)

        try:
            inserted_uuid = await col.data.insert(
                properties=props,
                uuid=id_point,
                vector=vec,
            )
            return PointStruct(
                id=str(inserted_uuid),
                payload={
                    "id": str(inserted_uuid),
                    "page_content": content,
                    "metadata": metadata or {},
                    "tenant_id": self.agent_id,
                },
                vector=vec,
            )
        except Exception as exc:
            log.error(f"Error inserting point into `{collection_name}`: {exc}")
            return None

    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        col = self._client.collections.get(collection_name)

        objects = []
        for p in points:
            payload = p.payload or {}
            payload["tenant_id"] = self.agent_id
            meta = payload.get("metadata", {})
            props = _meta_props(payload.get("page_content", ""), meta, self.agent_id)
            objects.append(
                wvc.data.DataObject(
                    properties=props,
                    uuid=str(p.id),
                    vector=list(p.vector) if p.vector else None,
                )
            )

        try:
            result = await col.data.insert_many(objects)
            status = "completed" if not result.has_errors else "partial"
            if result.has_errors:
                log.warning(
                    f"{len(result.errors)} objects failed during batch insert "
                    f"into `{collection_name}`."
                )
            return UpdateResult(status=status, operation_id=0)
        except Exception as exc:
            log.error(f"Batch insert error for `{collection_name}`: {exc}")
            return UpdateResult(status="error", operation_id=0)

    # ------------------------------------------------------------------
    # CRUD -- delete points
    # ------------------------------------------------------------------

    async def delete_tenant_points(
        self, collection_name: str, metadata: Dict | None = None
    ) -> UpdateResult:
        col = self._client.collections.get(collection_name)
        flt = self._build_filter(metadata)
        try:
            result = await col.data.delete_many(where=flt)
            status = "completed" if result.failed == 0 else "partial"
            return UpdateResult(status=status, operation_id=0)
        except Exception as exc:
            log.error(f"Error deleting points from `{collection_name}`: {exc}")
            return UpdateResult(status="error", operation_id=0)

    async def delete_tenant_points_by_ids(
        self, collection_name: str, points_ids: List
    ) -> UpdateResult:
        col = self._client.collections.get(collection_name)
        errors = 0
        for pid in points_ids:
            try:
                await col.data.delete_by_id(str(pid))
            except Exception as exc:
                log.warning(f"Could not delete `{pid}` from `{collection_name}`: {exc}")
                errors += 1
        return UpdateResult(
            status="completed" if errors == 0 else "partial",
            operation_id=0,
        )

    # ------------------------------------------------------------------
    # Point retrieval
    # ------------------------------------------------------------------

    async def retrieve_tenant_points(
        self, collection_name: str, points: List
    ) -> List[Record]:
        """
        Retrieve specific points by ID. All IDs must be valid UUID strings.
        """
        col = self._client.collections.get(collection_name)
        str_ids = [str(p) for p in points]
        flt = _combine(
            _tenant_filter(self.agent_id),
            WFilter.by_id().contains_any(str_ids),
        )
        result = await col.query.fetch_objects(
            filters=flt,
            limit=len(points),
            include_vector=True,
        )
        return [_obj_to_record(o) for o in result.objects]

    # ------------------------------------------------------------------
    # Scroll helpers
    # ------------------------------------------------------------------

    async def _scroll(
        self,
        collection_name: str,
        scroll_filter: WeaviateFilter,
        limit: int | None,
        offset: str | None,
        with_vectors: bool,
    ) -> Tuple[List[Record], str | None]:
        """
        Cursor-based pagination using Weaviate's ``after`` parameter.

        ``offset`` is a UUID string (the last UUID from the previous page) or
        ``None`` to start from the beginning.  Returns ``(records, next_cursor)``
        where ``next_cursor`` is ``None`` when there are no more pages.
        """
        col = self._client.collections.get(collection_name)

        if limit is not None:
            result = await col.query.fetch_objects(
                filters=scroll_filter,
                limit=limit,
                after=offset,
                include_vector=with_vectors,
            )
            records = [_obj_to_record(o) for o in result.objects]
            next_cursor = (
                str(result.objects[-1].uuid)
                if result.objects and len(result.objects) == limit
                else None
            )
            return records, next_cursor

        # No limit: paginate until exhausted.
        batch_size = 1000
        all_records: List[Record] = []
        cursor = offset
        while True:
            result = await col.query.fetch_objects(
                filters=scroll_filter,
                limit=batch_size,
                after=cursor,
                include_vector=with_vectors,
            )
            if not result.objects:
                break
            all_records.extend(_obj_to_record(o) for o in result.objects)
            if len(result.objects) < batch_size:
                break
            cursor = str(result.objects[-1].uuid)
        return all_records, None

    async def get_all_tenant_points(
        self,
        collection_name: str,
        limit: int | None = None,
        offset: str | None = None,
        metadata: Dict | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], str | None]:
        return await self._scroll(
            collection_name, self._build_filter(metadata), limit, offset, with_vectors
        )

    async def get_all_tenant_points_from_web(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], str | None]:
        flt = _combine(
            _tenant_filter(self.agent_id),
            WFilter.by_property("metadata__source").like("http*"),
        )
        return await self._scroll(collection_name, flt, limit, offset, with_vectors=False)

    async def get_all_tenant_points_from_files(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], str | None]:
        """
        Returns objects whose ``source`` is non-empty and does not start with
        ``http``.  Objects with no source at all (pure conversational memory)
        are excluded because they are not file-backed.

        Note: the Qdrant original has a latent bug here (MatchValue with a
        regex string does not actually perform regex matching in Qdrant).
        This implementation fixes the intent.
        """
        flt = _combine(
            _tenant_filter(self.agent_id),
            WFilter.by_property("metadata__source").not_equal(""),
            WFilter.not_(WFilter.by_property("metadata__source").like("http*")),
        )
        return await self._scroll(collection_name, flt, limit, offset, with_vectors=False)

    async def get_tenant_vectors_count(self, collection_name: str) -> int:
        col = self._client.collections.get(collection_name)
        agg = await col.aggregate.over_all(
            filters=_tenant_filter(self.agent_id),
            total_count=True,
        )
        return agg.total_count or 0

    # ------------------------------------------------------------------
    # Memory recall
    # ------------------------------------------------------------------

    async def recall_tenant_memory_from_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        metadata: Dict | None = None,
        k: int | None = 5,
        threshold: float | None = None,
    ) -> List[DocumentRecall]:
        col = self._client.collections.get(collection_name)
        flt = self._build_filter(metadata)

        result = await col.query.near_vector(
            near_vector=embedding,
            filters=flt,
            limit=k or 5,
            # Weaviate uses cosine *distance*; Qdrant threshold is cosine *similarity*.
            # max_distance = 1 - min_similarity
            distance=(1.0 - threshold) if threshold is not None else None,
            include_vector=True,
            return_metadata=MetadataQuery(distance=True),
        )

        return [
            _record_to_doc_recall(_obj_to_scored_point(obj, score=_distance_to_score(obj)))
            for obj in result.objects
        ]

    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        all_points, _ = await self.get_all_tenant_points(collection_name, with_vectors=True)
        return [_record_to_doc_recall(p) for p in all_points]

    # ------------------------------------------------------------------
    # Low-level search
    # ------------------------------------------------------------------

    async def search_in_tenant(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> List[ScoredPoint]:
        col = self._client.collections.get(collection_name)
        conditions: List[WeaviateFilter] = [_tenant_filter(self.agent_id)]
        if query_filter is not None:
            conditions.append(query_filter)
        flt = _combine(*conditions)

        result = await col.query.near_vector(
            near_vector=query_vector,
            filters=flt,
            limit=limit,
            distance=(1.0 - score_threshold) if score_threshold is not None else None,
            include_vector=with_vectors,
            return_metadata=MetadataQuery(distance=True),
        )

        return [
            _obj_to_scored_point(obj, score=_distance_to_score(obj))
            for obj in result.objects
        ]

    async def search_prefetched_in_tenant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        query_filter: Any,
        k: int,
        k_prefetched: int,
        threshold: float,
    ) -> List[ScoredPoint]:
        """
        Hybrid BM25+vector search using Weaviate's native ``hybrid()`` with
        Relative Score Fusion.  This replaces Qdrant's sparse-vector prefetch
        + RRF pattern; both achieve the same semantic goal of combining keyword
        and semantic signals.

        ``k_prefetched`` has no direct equivalent; Weaviate manages its
        internal candidate pool automatically.
        """
        col = self._client.collections.get(collection_name)
        conditions: List[WeaviateFilter] = [_tenant_filter(self.agent_id)]
        if query_filter is not None:
            conditions.append(query_filter)
        flt = _combine(*conditions)

        result = await col.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=0.5,                                         # equal BM25 / vector weight
            fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
            filters=flt,
            limit=k,
            include_vector=True,
            return_metadata=MetadataQuery(score=True),
        )

        scored: List[ScoredPoint] = []
        for obj in result.objects:
            score = (obj.metadata.score or 0.0) if obj.metadata else 0.0
            if score >= threshold:
                scored.append(_obj_to_scored_point(obj, score=score))
        return scored

    # ------------------------------------------------------------------
    # Filter builders (interface contract)
    # ------------------------------------------------------------------

    def build_condition(self, key: str, value: Any) -> List[WeaviateFilter]:
        """
        Recursively build Weaviate filter(s) from a key/value pair.

        Dot-notation keys (e.g. ``"source.url"``) are mapped to Weaviate's
        double-underscore property convention (``metadata__source__url``).
        """
        out: List[WeaviateFilter] = []

        if isinstance(value, dict):
            for k, v in value.items():
                out.extend(self.build_condition(f"{key}.{k}", v))
            return out

        if isinstance(value, list):
            for v in value:
                sub_key = f"{key}[]" if isinstance(v, dict) else key
                out.extend(self.build_condition(sub_key, v))
            return out

        weaviate_key = f"metadata__{key.replace('.', '__')}"
        out.append(WFilter.by_property(weaviate_key).equal(value))
        return out

    def filter_from_dict(self, filter_dict: Dict) -> WeaviateFilter | None:
        if not filter_dict:
            return None
        conditions = [
            cond
            for key, value in filter_dict.items()
            for cond in self.build_condition(key, value)
        ]
        if not conditions:
            return None
        return WFilter.any_of(conditions) if len(conditions) > 1 else conditions[0]

    # ------------------------------------------------------------------
    # Snapshots / dumps
    # ------------------------------------------------------------------

    async def save_dump(self, collection_name: str, folder: str = "dormouse/"):
        """
        Trigger a Weaviate filesystem backup for the specified collection.

        Requires the Weaviate server to be started with:
            ENABLE_MODULES=backup-filesystem
            BACKUP_FILESYSTEM_PATH=/some/path

        ``folder`` is accepted for API compatibility but the physical path is
        controlled by the server-side ``BACKUP_FILESYSTEM_PATH`` setting.
        """
        if not self.save_memory_snapshots:
            return
        backup_id = f"cat-{collection_name.lower()}-{uuid.uuid4().hex[:8]}"
        try:
            await self._client.backup.create(
                backup_id=backup_id,
                backend=BackupStorage.FILESYSTEM,
                include_collections=[collection_name],
                wait_for_completion=True,
            )
            log.warning(
                f"Backup `{backup_id}` for collection `{collection_name}` "
                f"(agent `{self.agent_id}`) completed."
            )
        except Exception as exc:
            log.warning(f"Could not save dump for `{collection_name}`: {exc}")


# ---------------------------------------------------------------------------
# Config model  (mirrors QdrantConfig)
# ---------------------------------------------------------------------------

class WeaviateConfig(VectorDatabaseSettings):
    host: str = get_env("CAT_WEAVIATE_HOST") or "localhost"
    http_port: int = 8080
    grpc_port: int = 50051
    api_key: str | None = get_env("CAT_WEAVIATE_API_KEY") or None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "humanReadableName": "Weaviate Vector Database",
            "description": "Configuration for a Weaviate vector database instance",
            "link": "https://weaviate.io",
        },
    )

    @classmethod
    def pyclass(cls) -> Type[WeaviateHandler]:
        return WeaviateHandler
