def check_username_existence(db, username):
    users_ref = db.reference('users')
    users = users_ref.get()

    if users:
        for user_id, user_data in users.items():
            if 'username' in user_data and user_data['username'] == username:
                return True  # Username already exists
    return False  # Username does not exist

def add_user_to_database(db, user_id, username, password):
    ref = db.reference('users/' + user_id)
    ref.set({
        'username': username,
        'password': password
    })

# Ví dụ về cách sử dụng hàm add_user_to_database
def connect(db, user_id, username, password):
    if not check_username_existence(db, username):
        add_user_to_database(db, user_id, username, password)
        return("User added to database successfully!")
    else:
        return("Username already exists in the database!")




