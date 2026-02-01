#!/usr/bin/env python3
"""Test that empty md-content clears the database."""
from ingest import DocumentIngester, MD_CONTENT_DIR
from pathlib import Path

# Use actual md-content directory
test_file = MD_CONTENT_DIR / 'test_cleanup.md'

# Ingest with custom table
ingester = DocumentIngester(table_name='test_cleanup_empty')

# Step 1: Create and ingest a test file
print('Step 1: Creating and ingesting test file...')
test_file.write_text('Test content for cleanup verification')
chunks = ingester.ingest_file(test_file)
print(f'✓ Created {chunks} chunks')

if chunks > 0:
    table = ingester.table.to_arrow()
    print(f'✓ Database has {len(table)} chunks\n')
    
    # Step 2: Delete the file
    test_file.unlink()
    print('Step 2: File deleted from md-content\n')
    
    # Step 3: Run cleanup with empty list
    print('Step 3: Running cleanup_orphaned_chunks with empty list...')
    deleted = ingester.cleanup_orphaned_chunks([])
    print(f'✓ Deleted {deleted} orphaned chunks\n')
    
    # Check result
    table = ingester.table.to_arrow()
    print(f'Result: Database has {len(table)} chunks')
    print(f'✓ SUCCESS!' if len(table) == 0 else f'✗ FAILED - still has {len(table)} chunks')
else:
    print('✗ Failed to create test chunks')

# Cleanup
if test_file.exists():
    test_file.unlink()
ingester.db.drop_table('test_cleanup_empty')
print('\n✓ Cleanup complete')
