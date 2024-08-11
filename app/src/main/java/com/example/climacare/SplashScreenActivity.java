package com.example.climacare;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;

public class SplashScreenActivity extends AppCompatActivity {

    private static final int SPLASH_DISPLAY_DURATION = 3000; // Splash screen duration in milliseconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.splash_screen);

        // Using a handler to delay the loading of LoginActivity
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Start LoginActivity after the splash screen duration
                Intent loginIntent = new Intent(SplashScreenActivity.this, LoginActivity.class);
                startActivity(loginIntent);
                finish(); // Close the splash screen activity to prevent it from appearing on back press
            }
        }, SPLASH_DISPLAY_DURATION);
    }
}
