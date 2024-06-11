def get_all_users_from_database(db):
    ref = db.reference('users')
    return ref.get()

# Ví dụ về cách sử dụng hàm get_all_users_from_database
def retrieve(db, username, password):
    all_users = get_all_users_from_database(db)
    if all_users:
        print("All Users:")
        for user_id, user_info in all_users.items():
            print("User ID:", user_id)
            print("Username:", user_info.get("username"))
            print("Password:", user_info.get("password"))
            print()
            if(user_info.get("username")==username and user_info.get("password")==password):
                return True
            
    return False

