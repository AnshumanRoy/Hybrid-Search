"""Document CRUD, sharing and access-control orchestration.

Access model:
- ``none``          → not visible (same as not existing).
- ``viewer``        → document public, shared as viewer.
- ``editor``        → owner (full) or shared as editor (re-index/update).
- ``owner``         → full control: update, delete, share, unshare.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.application.services.ingestion import IngestionService
from app.domain.entities import (
    DocumentAccess,
    DocumentRecord,
    IngestionResult,
    ShareRecord,
    SharingPermission,
)
from app.domain.exceptions import DocumentNotFoundError, PermissionDeniedError, UserNotFoundError
from app.domain.interfaces import DocumentRepository, UserRepository

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    """Result of an ingest: the ingestion result plus access to the affected
    document (``None`` when the document is not accessible to the caller)."""

    ingestion: IngestionResult
    access: DocumentAccess | None = None


@dataclass
class UpdateOutcome:
    access: DocumentAccess
    ingestion: IngestionResult


class DocumentService:
    def __init__(
        self,
        ingestion: IngestionService,
        repository: DocumentRepository,
        users: UserRepository,
    ) -> None:
        self.ingestion = ingestion
        self.repository = repository
        self.users = users

    # --- listing / access ---

    def list(self, user_id: str) -> list[DocumentAccess]:
        docs = self.repository.list_accessible_documents(user_id)
        user_email = self._user_email(user_id)
        return [access for doc in docs if (access := self._access_for(user_id, doc, user_email))]

    def get(self, user_id: str, document_id: str) -> DocumentAccess:
        doc = self.repository.get_document(document_id)
        access = self._access_for(user_id, doc, self._user_email(user_id))
        if access is None:
            raise DocumentNotFoundError(document_id)
        return access

    def accessible_document_ids(self, user_id: str) -> set[str]:
        return self.repository.list_accessible_document_ids(user_id)

    # --- CRUD ---

    def ingest(self, user_id: str, filename: str, data: bytes) -> IngestOutcome:
        user_email = self._user_email(user_id)
        result = self.ingestion.process_bytes(filename, data, owner_id=user_id)
        doc = self.repository.get_document(result.document_id) if result.document_id else None
        access = self._access_for(user_id, doc, user_email)
        return IngestOutcome(ingestion=result, access=access)

    def update(self, user_id: str, document_id: str, filename: str, data: bytes) -> UpdateOutcome:
        existing = self.repository.get_document(document_id)
        user_email = self._user_email(user_id)
        access = self._access_for(user_id, existing, user_email)
        if access is None:
            raise DocumentNotFoundError(document_id)
        if access.permission is not SharingPermission.EDITOR:
            raise PermissionDeniedError(f"no edit permission on document {document_id!r}")
        # editors may re-index; ownership stays with the original owner.
        result = self.ingestion.process_bytes(filename, data, owner_id=existing.owner_id)
        if result.status == "indexed" and result.document_id != document_id:
            self._migrate_shares_to_new_document(user_id, document_id, result.document_id)
            self.ingestion.persister.delete_document(document_id)
            new_doc = self.repository.get_document(result.document_id)
            return UpdateOutcome(
                access=self._require_access(user_id, new_doc, user_email),
                ingestion=result,
            )
        fresh = self.repository.get_document(result.document_id) or existing
        return UpdateOutcome(access=self._require_access(user_id, fresh, user_email), ingestion=result)

    def delete(self, user_id: str, document_id: str) -> None:
        existing = self.repository.get_document(document_id)
        access = self._access_for(user_id, existing, self._user_email(user_id))
        if access is None:
            raise DocumentNotFoundError(document_id)
        if not self._is_owner(user_id, existing):
            raise PermissionDeniedError(f"only the owner can delete document {document_id!r}")
        self.ingestion.persister.delete_document(document_id)
        logger.info("document deleted; document_id=%s by=%s", document_id, user_id)

    # --- sharing ---

    def share(
        self,
        user_id: str,
        document_id: str,
        user_email: str,
        permission: SharingPermission,
    ) -> ShareRecord:
        existing = self.repository.get_document(document_id)
        if self._access_for(user_id, existing, self._user_email(user_id)) is None:
            raise DocumentNotFoundError(document_id)
        if not self._is_owner(user_id, existing):
            raise PermissionDeniedError(f"only the owner can share document {document_id!r}")
        if existing.owner_id is None:
            raise PermissionDeniedError("public documents have no owner and cannot be shared")
        if self._user_email(user_id).lower() == user_email.strip().lower():
            raise ValueError("cannot share a document with yourself")
        share = self.repository.share_document(document_id, user_email.strip().lower(), permission)
        if share is None:
            raise UserNotFoundError(f"no user with email {user_email!r} exists")
        return share

    def unshare(self, user_id: str, document_id: str, user_email: str) -> None:
        existing = self.repository.get_document(document_id)
        if self._access_for(user_id, existing, self._user_email(user_id)) is None:
            raise DocumentNotFoundError(document_id)
        if not self._is_owner(user_id, existing):
            raise PermissionDeniedError(f"only the owner can unshare document {document_id!r}")
        self.repository.unshare_document(document_id, user_email.strip().lower())

    def list_shares(self, user_id: str, document_id: str) -> list[ShareRecord]:
        existing = self.repository.get_document(document_id)
        if self._access_for(user_id, existing, self._user_email(user_id)) is None:
            raise DocumentNotFoundError(document_id)
        if not self._is_owner(user_id, existing):
            raise PermissionDeniedError(f"only the owner can list shares for document {document_id!r}")
        return self.repository.list_shares(document_id)

    # --- internal ---

    def _access_for(
        self,
        user_id: str,
        doc: DocumentRecord | None,
        user_email: str | None = None,
    ) -> DocumentAccess | None:
        if doc is None:
            return None
        if doc.owner_id == user_id:
            return DocumentAccess(document=doc, permission=SharingPermission.EDITOR, shared=False)
        if doc.owner_id is None:
            return DocumentAccess(document=doc, permission=SharingPermission.VIEWER, shared=False)
        email = user_email or self._user_email(user_id)
        permission = self.repository.get_shared_permission(doc.document_id, email)
        if permission is None:
            return None
        return DocumentAccess(document=doc, permission=permission, shared=True)

    def _require_access(self, user_id: str, doc: DocumentRecord | None, user_email: str) -> DocumentAccess:
        access = self._access_for(user_id, doc, user_email)
        if access is None:
            raise DocumentNotFoundError()
        return access

    @staticmethod
    def _is_owner(user_id: str, doc: DocumentRecord) -> bool:
        return doc.owner_id == user_id

    def _user_email(self, user_id: str) -> str | None:
        user = self.users.get_user(user_id)
        return user.email if user else None

    def _migrate_shares_to_new_document(self, owner_id: str, old_document_id: str, new_document_id: str) -> None:
        for share in self.repository.list_shares(old_document_id):
            self.repository.share_document(new_document_id, share.shared_with_email, share.permission)