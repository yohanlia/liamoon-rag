from supabase import create_client, Client
from typing import Any, Dict, List, Optional
from config import SUPABASE_URL, SUPABASE_KEY

class SupabaseHandler:
    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        """
        Initialize a Supabase client.
        Args:
            url (str): The Supabase project URL.
            key (str): The Supabase API key (service_role or anon key).
        """

        self.url = url

        self.key = key

        self.client: Client = create_client(url, key)

    # ----------------------------
    # CRUD Operations
    # ----------------------------

    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new row into a table."""

        response = self.client.table(table).insert(data).execute()
        return response.data

    def select(self, table: str, columns: str = "*", filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Select rows from a table with optional filters."""

        query = self.client.table(table).select(columns)
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        response = query.execute()
        return response.data

    def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update rows in a table based on filters."""

        query = self.client.table(table).update(updates)
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data

    def delete(self, table: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete rows from a table based on filters."""

        query = self.client.table(table).delete()
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data

    def vector_search(self, embedding, match_count=5):
        response = (
            self.client.rpc(
                "match_document_chunks",
                {
                    "query_embedding": embedding,
                    "match_count": match_count
                }
            ).execute()
        )
        return response.data