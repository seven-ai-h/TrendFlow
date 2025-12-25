print("Testing imports...")

try:
    print("Importing Story...")
    from database.models import Story
    print("Story imported successfully!")
except Exception as e:
    print(f"Error importing Story: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing db functions...")
    from database.db_setup import db_connection, getSession
    print("DB functions imported successfully!")
except Exception as e:
    print(f"Error importing db functions: {e}")
    import traceback
    traceback.print_exc()

print("All imports complete!")