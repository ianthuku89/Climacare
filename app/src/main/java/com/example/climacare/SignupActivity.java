package com.example.climacare;

import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class SignupActivity extends AppCompatActivity {

    private EditText signupUsernameEditText;
    private EditText signupPasswordEditText;
    private Button signupButton;
    private DatabaseHelper dbHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signup);

        signupUsernameEditText = findViewById(R.id.signupUsernameEditText);
        signupPasswordEditText = findViewById(R.id.signupPasswordEditText);
        signupButton = findViewById(R.id.signupButton);

        dbHelper = new DatabaseHelper(this);

        signupButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String username = signupUsernameEditText.getText().toString();
                String password = signupPasswordEditText.getText().toString();

                if (!username.isEmpty() && !password.isEmpty()) {
                    if (!isUsernameTaken(username)) {
                        SQLiteDatabase db = dbHelper.getWritableDatabase();
                        ContentValues values = new ContentValues();
                        values.put("username", username);
                        values.put("password", password);
                        long newRowId = db.insert("users", null, values);
                        if (newRowId != -1) {
                            Toast.makeText(SignupActivity.this, "Signup successful!", Toast.LENGTH_SHORT).show();
                            Intent intent = new Intent(SignupActivity.this, LoginActivity.class);
                            startActivity(intent);
                            finish();
                        } else {
                            Toast.makeText(SignupActivity.this, "Signup failed! Please try again.", Toast.LENGTH_SHORT).show();
                        }
                    } else {
                        Toast.makeText(SignupActivity.this, "Username already taken! Please choose a different username.", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(SignupActivity.this, "Please fill in all fields", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private boolean isUsernameTaken(String username) {
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor = db.query(
                "users",
                new String[]{"username"},
                "username = ?",
                new String[]{username},
                null,
                null,
                null
        );

        boolean isTaken = cursor.getCount() > 0;
        cursor.close();
        return isTaken;
    }
}
