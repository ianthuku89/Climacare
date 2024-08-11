package com.example.climacare;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class BaseActivity extends AppCompatActivity {

    protected ImageButton historyButton;
    protected ImageButton chatBotButton;
    protected ImageButton callPageButton;
    protected ImageButton homeButton;
    protected ImageButton forecastButton;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public void setContentView(int layoutResID) {
        ViewGroup fullLayout = (ViewGroup) getLayoutInflater().inflate(R.layout.activity_base, null);
        FrameLayout activityContainer = fullLayout.findViewById(R.id.activity_content);
        getLayoutInflater().inflate(layoutResID, activityContainer, true);
        super.setContentView(fullLayout);

        // Initialize footer buttons
        historyButton = findViewById(R.id.historyButton);
        chatBotButton = findViewById(R.id.chatBotButton);
        callPageButton = findViewById(R.id.callPageButton);
        homeButton = findViewById(R.id.homeButton);
        forecastButton = findViewById(R.id.forecastButton);

        // Set up listeners
        historyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(BaseActivity.this, PredictionHistoryActivity.class));
            }
        });

        chatBotButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(BaseActivity.this, ChatbotActivity.class));
            }
        });

        callPageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(BaseActivity.this, CallPageActivity.class));
            }
        });

        homeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(BaseActivity.this, MainActivity.class);
                String username = getUsername();
                if (username != null) {
                    intent.putExtra("username", username);
                }
                startActivity(intent);
            }
        });

        forecastButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(BaseActivity.this, forecast.class));
            }
        });
    }

    protected String getUsername() {
        SharedPreferences prefs = getSharedPreferences("MyPrefs", MODE_PRIVATE);
        return prefs.getString("username", "User");
    }
}
