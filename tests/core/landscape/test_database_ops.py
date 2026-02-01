"""Tests for database operation helpers."""

from unittest.mock import MagicMock

from elspeth.core.landscape._database_ops import DatabaseOps


class TestDatabaseOps:
    """Test database operation helper methods."""

    def test_execute_fetchone_returns_row(self) -> None:
        """execute_fetchone returns single row from query."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock(id="row1")

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchone.return_value = mock_row

        ops = DatabaseOps(mock_db)
        query = MagicMock()

        result = ops.execute_fetchone(query)

        assert result == mock_row
        mock_conn.execute.assert_called_once_with(query)

    def test_execute_fetchone_returns_none_when_no_row(self) -> None:
        """execute_fetchone returns None when no row found."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchone.return_value = None

        ops = DatabaseOps(mock_db)
        result = ops.execute_fetchone(MagicMock())

        assert result is None

    def test_execute_fetchall_returns_list(self) -> None:
        """execute_fetchall returns list of rows."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_rows = [MagicMock(id="row1"), MagicMock(id="row2")]

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_result.fetchall.return_value = mock_rows

        ops = DatabaseOps(mock_db)
        result = ops.execute_fetchall(MagicMock())

        assert result == mock_rows

    def test_execute_insert_commits(self) -> None:
        """execute_insert executes insert statement."""
        mock_db = MagicMock()
        mock_conn = MagicMock()

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

        ops = DatabaseOps(mock_db)
        stmt = MagicMock()

        ops.execute_insert(stmt)

        mock_conn.execute.assert_called_once_with(stmt)

    def test_execute_update_commits(self) -> None:
        """execute_update executes update statement."""
        mock_db = MagicMock()
        mock_conn = MagicMock()

        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

        ops = DatabaseOps(mock_db)
        stmt = MagicMock()

        ops.execute_update(stmt)

        mock_conn.execute.assert_called_once_with(stmt)
