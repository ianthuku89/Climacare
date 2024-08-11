package com.example.climacare;

import android.provider.BaseColumns;

public final class UserContract {

    private UserContract() {}

    public static final class UserEntry implements BaseColumns {
        public static final String TABLE_NAME = "users";
        public static final String COLUMN_USERNAME = "username";
        public static final String COLUMN_PASSWORD = "password";
    }
}
